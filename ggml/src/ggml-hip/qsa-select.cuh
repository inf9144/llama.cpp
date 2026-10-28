#pragma once

#include <cstddef>
#include <cstdint>
#include <climits>
#include <cmath>

struct qsa_select_params {
    const char * top;
    const char * score;
    const char * pools;
    const char * tail;
    int32_t * dst;
    int n_top, n_pool, n_sel, pool_size, n_kv, sort_size;
    size_t t0, t1, s0, s1, p0, p1, e0, e1;
};

#if __has_include(<hipcub/block/block_radix_sort.hpp>) && __has_include(<rocprim/block/block_radix_sort.hpp>)
#include <hipcub/block/block_radix_sort.hpp>
#define GGML_HIP_QSA_RADIX_AVAILABLE

static __device__ __forceinline__ int qsa_radix_bound(const int32_t * values, int begin, int end, int32_t value, bool upper) {
    while (begin < end) {
        const int mid = begin + (end - begin)/2;
        if (values[mid] < value || (upper && values[mid] == value)) {
            begin = mid + 1;
        } else {
            end = mid;
        }
    }
    return begin;
}

static __device__ __forceinline__ void qsa_radix_swap(int32_t & a, int32_t & b) {
    if (a > b) {
        const int32_t t = a;
        a = b;
        b = t;
    }
}

static __device__ void qsa_radix_gather_members(const qsa_select_params p, int members_size, int32_t * values) {
    const int row = blockIdx.x;
    for (int slot = threadIdx.x; slot < p.n_top; slot += 256) {
        const int pool = *(const int32_t *) (p.top + row*p.t1 + slot*p.t0);
        const bool live = pool >= 0 && pool < p.n_pool &&
            *(const float *) (p.score + row*p.s1 + pool*p.s0) > -INFINITY;
        int32_t cells[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int cell = live ? *(const int32_t *) (p.pools + pool*p.p1 + j*p.p0) : -1;
            cells[j] = cell >= 0 && cell < p.n_kv ? cell : INT32_MAX;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            values[slot*4 + j] = cells[j];
        }
    }
    for (int i = 4*p.n_top + threadIdx.x; i < members_size; i += 256) {
        values[i] = INT32_MAX;
    }
    __syncthreads();
}

static __device__ void qsa_radix_finish(const qsa_select_params p, const int32_t * sorted, int32_t * merged, int members_size, int * warp_offsets) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid%32;
    const int warp = tid/32;
    int32_t tail[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int cell = *(const int32_t *) (p.tail + row*p.e1 + j*p.e0);
        tail[j] = cell >= 0 && cell < p.n_kv ? cell : INT32_MAX;
    }
    qsa_radix_swap(tail[0], tail[1]);
    qsa_radix_swap(tail[1], tail[2]);
    qsa_radix_swap(tail[0], tail[1]);
    for (int i = tid; i < members_size; i += 256) {
        const int32_t value = sorted[i];
        const int rank = (tail[0] < value) + (tail[1] < value) + (tail[2] < value);
        merged[i + rank] = value;
    }
    if (tid < 3) {
        const int rank = qsa_radix_bound(sorted, 0, members_size, tail[tid], true);
        merged[rank + tid] = tail[tid];
    }
    __syncthreads();

    const int size = members_size + 3;
    const int begin = tid*size/256;
    const int end = (tid + 1)*size/256;
    int count = 0;
    for (int i = begin; i < end; ++i) {
        count += merged[i] != INT32_MAX && (i == 0 || merged[i] != merged[i - 1]);
    }
    int32_t * out = p.dst + int64_t(row)*p.n_sel;
    for (int i = tid; i < p.n_sel; i += 256) { out[i] = -1; }
    int sum = count;
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
        const int add = __shfl_up(sum, d, 32);
        if (lane >= d) { sum += add; }
    }
    if (lane == 31) { warp_offsets[warp] = sum; }
    __syncthreads();
    const int local_offset = sum - count;
    if (warp == 0) {
        const int total = lane < 8 ? warp_offsets[lane] : 0;
        int prefix = total;
#pragma unroll
        for (int d = 1; d < 32; d *= 2) {
            const int add = __shfl_up(prefix, d, 32);
            if (lane >= d) { prefix += add; }
        }
        if (lane < 8) { warp_offsets[lane] = prefix - total; }
    }
    __syncthreads();
    int offset = warp_offsets[warp] + local_offset;
    for (int i = begin; i < end; ++i) {
        if (merged[i] != INT32_MAX && (i == 0 || merged[i] != merged[i - 1])) {
            out[offset++] = merged[i];
        }
    }
}

template <int members_size>
__launch_bounds__(256)
static __global__ void qsa_select_radix_kernel(const qsa_select_params p) {
    constexpr int items = members_size/256;
    using sorter = hipcub::BlockRadixSort<uint32_t, 256, items>;
    __shared__ union {
        typename sorter::TempStorage sort;
        int32_t values[members_size + 3];
    } storage;
    __shared__ int32_t merged[members_size + 3];
    __shared__ int32_t offsets[8];
    qsa_radix_gather_members(p, members_size, storage.values);
    uint32_t keys[items];
#pragma unroll
    for (int i = 0; i < items; ++i) {
        const int value = storage.values[threadIdx.x*items + i];
        keys[i] = value == INT32_MAX ? p.n_kv : value;
    }
    __syncthreads();
    // n_kv sorts after every valid cell; sorting only its significant bits is sufficient.
    sorter(storage.sort).Sort(keys, 0, 32 - __clz((uint32_t) p.n_kv));
    __syncthreads();
#pragma unroll
    for (int i = 0; i < items; ++i) {
        storage.values[threadIdx.x*items + i] = keys[i] >= uint32_t(p.n_kv) ? INT32_MAX : keys[i];
    }
    __syncthreads();
    qsa_radix_finish(p, storage.values, merged, members_size, offsets);
}

#endif // GGML_HIP_QSA_RADIX_AVAILABLE
