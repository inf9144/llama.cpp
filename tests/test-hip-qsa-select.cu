#include "qsa-select-common.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <hip/hip_runtime.h>
#if __has_include(<hipcub/block/block_radix_sort.hpp>) && __has_include(<rocprim/block/block_radix_sort.hpp>)
#include <hipcub/block/block_radix_sort.hpp>
#define QSA_TEST_HIPCUB
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <iterator>
#include <memory>
#include <vector>

#define QSA_HIP_CHECK(expr) do { const hipError_t err = (expr); if (err != hipSuccess) { GGML_ABORT("%s: %s", #expr, hipGetErrorString(err)); } } while (0)

namespace qsa_test {

// Frozen e7b431d selection kernel. The suite also compares it to the actual backend op.
__launch_bounds__(256)
static __global__ void select_baseline(const params p) {
    extern __shared__ int32_t shared[];
    int32_t * values = shared;
    int32_t * offsets = shared + p.sort_size;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_members = p.n_top*p.pool_size;
    int32_t * out = p.dst + int64_t(row)*p.n_sel;

    for (int i = tid; i < p.sort_size; i += 256) {
        int cell = -1;
        if (i < n_members) {
            const int pool = *(const int32_t *) (p.top + row*p.t1 + (i/p.pool_size)*p.t0);
            if (pool >= 0 && pool < p.n_pool &&
                    *(const float *) (p.score + row*p.s1 + pool*p.s0) > -INFINITY) {
                cell = *(const int32_t *) (p.pools + pool*p.p1 + (i%p.pool_size)*p.p0);
            }
        } else if (i < p.n_sel) {
            cell = *(const int32_t *) (p.tail + row*p.e1 + (i - n_members)*p.e0);
        }
        values[i] = cell >= 0 && cell < p.n_kv ? cell : INT32_MAX;
    }
    __syncthreads();

    for (int k = 2; k <= p.sort_size; k *= 2) {
        for (int j = k/2; j > 0; j /= 2) {
            for (int i = tid; i < p.sort_size; i += 256) {
                const int other = i ^ j;
                if (other > i) {
                    const int a = values[i];
                    const int b = values[other];
                    if ((i & k) == 0 ? a > b : a < b) {
                        values[i] = b;
                        values[other] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    const int begin = tid*p.sort_size/256;
    const int end = (tid + 1)*p.sort_size/256;
    int count = 0;
    for (int i = begin; i < end; ++i) {
        count += values[i] != INT32_MAX && (i == 0 || values[i] != values[i - 1]);
    }
    offsets[tid] = count;
    for (int i = tid; i < p.n_sel; i += 256) {
        out[i] = -1;
    }
    __syncthreads();
    for (int step = 1; step < 256; step *= 2) {
        const int add = tid >= step ? offsets[tid - step] : 0;
        __syncthreads();
        offsets[tid] += add;
        __syncthreads();
    }
    int offset = offsets[tid] - count;
    for (int i = begin; i < end; ++i) {
        if (values[i] != INT32_MAX && (i == 0 || values[i] != values[i - 1])) {
            out[offset++] = values[i];
        }
    }
}

template <bool presort>
static __device__ void gather_members(const params p, int members_size, int32_t * values) {
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
        if constexpr (presort) {
            sort4(cells[0], cells[1], cells[2], cells[3]);
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

static __device__ void finish(const params p, const int32_t * sorted, int32_t * merged, int members_size, int * warp_offsets) {
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
    swap_ordered(tail[0], tail[1]);
    swap_ordered(tail[1], tail[2]);
    swap_ordered(tail[0], tail[1]);
    for (int i = tid; i < members_size; i += 256) {
        const int32_t value = sorted[i];
        const int rank = (tail[0] < value) + (tail[1] < value) + (tail[2] < value);
        merged[i + rank] = value;
    }
    if (tid < 3) {
        const int rank = bound(sorted, 0, members_size, tail[tid], true);
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

template <bool merge>
__launch_bounds__(256)
static __global__ void select_members(const params p, int members_size) {
    extern __shared__ int32_t shared[];
    int32_t * a = shared;
    int32_t * b = a + members_size + 3;
    int32_t * offsets = b + members_size + 3;
    gather_members<merge>(p, members_size, a);
    if constexpr (merge) {
        for (int run = 4; run < members_size; run *= 2) {
            for (int i = threadIdx.x; i < members_size; i += 256) {
                const int group = i/(2*run)*(2*run);
                const bool right = i - group >= run;
                const int other = group + (right ? 0 : run);
                const int rank = bound(a, other, other + run, a[i], right) - other;
                b[group + (i - group)%run + rank] = a[i];
            }
            __syncthreads();
            int32_t * temp = a; a = b; b = temp;
        }
    } else {
        for (int k = 2; k <= members_size; k *= 2) {
            for (int j = k/2; j > 0; j /= 2) {
                for (int i = threadIdx.x; i < members_size; i += 256) {
                    const int other = i ^ j;
                    if (other > i) {
                        const int x = a[i], y = a[other];
                        if ((i & k) == 0 ? x > y : x < y) { a[i] = y; a[other] = x; }
                    }
                }
                __syncthreads();
            }
        }
    }
    finish(p, a, b, members_size, offsets);
}

#ifdef QSA_TEST_HIPCUB
template <int members_size>
__launch_bounds__(256)
static __global__ void select_radix(const params p) {
    constexpr int items = members_size/256;
    using sorter = hipcub::BlockRadixSort<uint32_t, 256, items>;
    __shared__ union {
        typename sorter::TempStorage sort;
        int32_t values[members_size + 3];
    } storage;
    __shared__ int32_t merged[members_size + 3];
    __shared__ int32_t offsets[8];
    gather_members<false>(p, members_size, storage.values);
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
    finish(p, storage.values, merged, members_size, offsets);
}
#endif

static const char * names[] = {"e7b431d-bitonic", "members-bitonic-tail", "pool-merge-tail",
#ifdef QSA_TEST_HIPCUB
    "hipcub-radix-tail",
#endif
};

static void launch(int candidate, const params p, int rows, hipStream_t stream) {
    int members_size = 256;
    while (members_size < p.n_top*4) { members_size *= 2; }
    const size_t bytes = (2*(members_size + 3) + 8)*sizeof(int32_t);
    switch (candidate) {
        case 0:
            hipLaunchKernelGGL(select_baseline, dim3(rows), dim3(256), (p.sort_size + 256)*sizeof(int32_t), stream, p);
            break;
        case 1:
            hipLaunchKernelGGL((select_members<false>), dim3(rows), dim3(256), bytes, stream, p, members_size);
            break;
        case 2:
            hipLaunchKernelGGL((select_members<true>), dim3(rows), dim3(256), bytes, stream, p, members_size);
            break;
#ifdef QSA_TEST_HIPCUB
        case 3:
            switch (members_size) {
#define QSA_RADIX_CASE(n) case n: hipLaunchKernelGGL((select_radix<n>), dim3(rows), dim3(256), 0, stream, p); break
                QSA_RADIX_CASE(256);
                QSA_RADIX_CASE(512);
                QSA_RADIX_CASE(1024);
                QSA_RADIX_CASE(2048);
                QSA_RADIX_CASE(4096);
#undef QSA_RADIX_CASE
                default: GGML_ABORT("invalid radix tile");
            }
            break;
#endif
        default: GGML_ABORT("invalid candidate");
    }
    QSA_HIP_CHECK(hipGetLastError());
}

struct graph_timer {
    hipStream_t stream;
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    hipEvent_t begin = nullptr, end = nullptr;
    static constexpr int repeats = 32;
    graph_timer(int candidate, const params p, int rows, hipStream_t stream) : stream(stream) {
        QSA_HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        for (int i = 0; i < repeats; ++i) { launch(candidate, p, rows, stream); }
        QSA_HIP_CHECK(hipStreamEndCapture(stream, &graph));
        QSA_HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        QSA_HIP_CHECK(hipEventCreate(&begin));
        QSA_HIP_CHECK(hipEventCreate(&end));
    }
    ~graph_timer() {
        QSA_HIP_CHECK(hipEventDestroy(end));
        QSA_HIP_CHECK(hipEventDestroy(begin));
        QSA_HIP_CHECK(hipGraphExecDestroy(exec));
        QSA_HIP_CHECK(hipGraphDestroy(graph));
    }
    std::array<double, 2> sample() {
        QSA_HIP_CHECK(hipGraphLaunch(exec, stream));
        QSA_HIP_CHECK(hipStreamSynchronize(stream));
        const auto start = std::chrono::steady_clock::now();
        QSA_HIP_CHECK(hipEventRecord(begin, stream));
        QSA_HIP_CHECK(hipGraphLaunch(exec, stream));
        QSA_HIP_CHECK(hipEventRecord(end, stream));
        QSA_HIP_CHECK(hipEventSynchronize(end));
        const double wall = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count()/repeats;
        float milliseconds;
        QSA_HIP_CHECK(hipEventElapsedTime(&milliseconds, begin, end));
        return {1000.0*milliseconds/repeats, wall};
    }
};
static double quantile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    return values[std::min(values.size() - 1, size_t(std::ceil(q*values.size())) - 1)];
}

template <typename T>
static void set_rows(ggml_tensor * t, const std::vector<T> & values) {
    std::vector<uint8_t> bytes(ggml_nbytes(t), 0);
    for (int64_t r = 0; r < t->ne[1]; ++r) {
        for (int64_t c = 0; c < t->ne[0]; ++c) {
            memcpy(bytes.data() + r*t->nb[1] + c*t->nb[0], &values[r*t->ne[0] + c], sizeof(T));
        }
    }
    ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
}

static void run_case(ggml_backend_t backend, int kv, int rows, int top_count, int layout, bool strided, bool benchmark, bool & printed_device) {
    const int n_pool = kv/4;
    const int n_sel = top_count*4 + 3;
    ggml_context_ptr ctx(ggml_init({1024*1024, nullptr, true}));
    auto input = [&](ggml_type type, int cols, int count) {
        ggml_tensor * root = ggml_new_tensor_2d(ctx.get(), type, cols + (strided ? 5 : 0), count);
        return strided ? ggml_view_2d(ctx.get(), root, cols, count, root->nb[1], 2*root->nb[0]) : root;
    };
    auto * top = input(GGML_TYPE_I32, top_count, rows);
    auto * score = input(GGML_TYPE_F32, n_pool, rows);
    auto * pools = input(GGML_TYPE_I32, 4, n_pool);
    auto * tail = input(GGML_TYPE_I32, 3, rows);
    auto * selected = ggml_qsa_select(ctx.get(), top, score, pools, tail, kv);
    GGML_ASSERT(ggml_backend_supports_op(backend, selected));
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer);
    auto * gf = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(gf, selected);
    hipPointerAttribute_t attr;
    QSA_HIP_CHECK(hipPointerGetAttributes(&attr, selected->data));
    QSA_HIP_CHECK(hipSetDevice(attr.device));
    hipDeviceProp_t prop;
    QSA_HIP_CHECK(hipGetDeviceProperties(&prop, attr.device));
    GGML_ASSERT(prop.warpSize == 32 &&
        (strncmp(prop.gcnArchName, "gfx1100", 7) == 0 || strncmp(prop.gcnArchName, "gfx1101", 7) == 0));
    if (!printed_device) {
        int runtime;
        QSA_HIP_CHECK(hipRuntimeGetVersion(&runtime));
        printf("device=%s arch=%s HIP=%d backend=%s baseline=e7b431d candidates=%zu hipcub=%s\n", prop.name,
            prop.gcnArchName, runtime, ggml_backend_name(backend), std::size(names),
#ifdef QSA_TEST_HIPCUB
            "available"
#else
            "not installed (radix candidate skipped)"
#endif
        );
        printed_device = true;
    }
    std::mt19937 gen(0x515341 + rows + top_count);
    std::vector<int32_t> cells(kv);
    for (int i = 0; i < kv; ++i) { cells[i] = i; }
    if (layout != 0) { std::shuffle(cells.begin(), cells.end(), gen); }
    std::vector<int32_t> picks(top_count*rows), pool_data(kv), tails(3*rows);
    std::vector<float> scores(n_pool*rows);
    for (int p = 0; p < n_pool; ++p) {
        for (int j = 0; j < 4; ++j) {
            int cell = cells[p*4 + j];
            if (layout >= 2) {
                if (p%13 == 0 || (p%19 == 0 && j == 3)) { cell = kv; }
                else if (p%17 == 0 && j == 2) { cell = -1; }
                else if (p%7 == 0 && j == 3) { cell = cells[p*4]; }
                else if (p > 0 && p%11 == 0 && j == 1) { cell = cells[(p - 1)*4]; }
            }
            pool_data[p*4 + j] = cell;
        }
    }
    for (int row = 0; row < rows; ++row) {
        for (int p = 0; p < n_pool; ++p) {
            float value = float(p%19);
            if (layout == 3 || (layout == 2 && (p + row)%5 == 0)) { value = -INFINITY; }
            else if (layout == 2 && p%17 == 0) { value = NAN; }
            else if (layout == 2 && p%23 == 0) { value = INFINITY; }
            scores[row*n_pool + p] = value;
        }
        for (int i = 0; i < top_count; ++i) {
            int pick = (i*37 + row*13)%n_pool;
            if (layout >= 2) {
                pick = i%29 == 0 ? n_pool : i%31 == 0 ? -1 : ((i%127)*37 + row*13)%n_pool;
            }
            picks[row*top_count + i] = pick;
        }
        tails[row*3 + 0] = layout == 3 ? -1 : cells[(row*13)%kv];
        tails[row*3 + 1] = layout == 3 ? kv + 7 : tails[row*3];
        tails[row*3 + 2] = kv;
    }
    set_rows(top, picks); set_rows(score, scores); set_rows(pools, pool_data); set_rows(tail, tails);
    std::vector<int32_t> expected;
    expected.reserve(size_t(rows)*n_sel);
    for (int row = 0; row < rows; ++row) {
        std::vector<int32_t> values;
        for (int i = 0; i < top_count; ++i) {
            const int p = picks[row*top_count + i];
            if (p < 0 || p >= n_pool || !(scores[row*n_pool + p] > -INFINITY)) { continue; }
            for (int j = 0; j < 4; ++j) {
                const int cell = pool_data[p*4 + j];
                if (cell >= 0 && cell < kv) { values.push_back(cell); }
            }
        }
        for (int j = 0; j < 3; ++j) {
            const int cell = tails[row*3 + j];
            if (cell >= 0 && cell < kv) { values.push_back(cell); }
        }
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        GGML_ASSERT(values.size() <= size_t(n_sel));
        values.resize(n_sel, -1);
        expected.insert(expected.end(), values.begin(), values.end());
    }
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    std::vector<int32_t> actual(expected.size());
    ggml_backend_tensor_get(selected, actual.data(), 0, ggml_nbytes(selected));
    if (actual != expected) {
        GGML_ABORT("backend reference mismatch: kv=%d rows=%d top=%d layout=%d strided=%d", kv, rows, top_count, layout, strided);
    }

    hipStream_t stream;
    QSA_HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    int32_t * output = nullptr;
    constexpr int guard = 32;
    const size_t output_bytes = (expected.size() + 2*guard)*sizeof(int32_t);
    QSA_HIP_CHECK(hipMalloc((void **) &output, output_bytes));
    int sort_size = 256;
    while (sort_size < n_sel) { sort_size *= 2; }
    const params p = {(const char *) top->data, (const char *) score->data, (const char *) pools->data,
        (const char *) tail->data, output + guard, top_count, n_pool, n_sel, 4, kv, sort_size,
        top->nb[0], top->nb[1], score->nb[0], score->nb[1], pools->nb[0], pools->nb[1], tail->nb[0], tail->nb[1]};
    auto check = [&](int candidate) {
        QSA_HIP_CHECK(hipStreamSynchronize(stream));
        std::vector<int32_t> all(expected.size() + 2*guard);
        QSA_HIP_CHECK(hipMemcpy(all.data(), output, output_bytes, hipMemcpyDeviceToHost));
        for (int i = 0; i < guard; ++i) {
            if (all[i] != 0x5a5a5a5a || all[guard + expected.size() + i] != 0x5a5a5a5a) {
                GGML_ABORT("guard overwrite: candidate=%s kv=%d rows=%d top=%d layout=%d strided=%d", names[candidate], kv, rows, top_count, layout, strided);
            }
        }
        const auto mismatch = std::mismatch(expected.begin(), expected.end(), all.begin() + guard);
        if (mismatch.first != expected.end()) {
            GGML_ABORT("selection mismatch: candidate=%s kv=%d rows=%d top=%d layout=%d strided=%d index=%zu expected=%d actual=%d",
                names[candidate], kv, rows, top_count, layout, strided, size_t(mismatch.first - expected.begin()), *mismatch.first, *mismatch.second);
        }
    };
    for (int candidate = 0; candidate < int(std::size(names)); ++candidate) {
        QSA_HIP_CHECK(hipMemsetAsync(output, 0x5a, output_bytes, stream));
        launch(candidate, p, rows, stream);
        check(candidate);
    }
    if (benchmark) {
        std::vector<std::unique_ptr<graph_timer>> timers;
        for (int c = 0; c < int(std::size(names)); ++c) {
            timers.emplace_back(new graph_timer(c, p, rows, stream));
            timers.back()->sample();
            check(c);
        }
        std::vector<std::vector<double>> gpu(std::size(names)), wall(std::size(names));
        for (int round = 0; round < 21; ++round) {
            for (int j = 0; j < int(timers.size()); ++j) {
                const int c = (j + round)%timers.size();
                const auto sample = timers[c]->sample();
                gpu[c].push_back(sample[0]); wall[c].push_back(sample[1]);
            }
        }
        const double baseline = quantile(gpu[0], 0.5);
        for (int c = 0; c < int(timers.size()); ++c) {
            printf("SELECT backend=%s kv=%d rows=%d layout=%s candidate=%s gpu_median_us=%.3f gpu_p95_us=%.3f graph_wall_us=%.3f speedup=%.3f exact=1 guard=1\n",
                ggml_backend_name(backend), kv, rows, layout == 0 ? "linear" : "permuted", names[c], quantile(gpu[c], 0.5),
                quantile(gpu[c], 0.95), quantile(wall[c], 0.5), baseline/quantile(gpu[c], 0.5));
        }
        std::vector<double> backend_wall;
        for (int round = 0; round < 7; ++round) {
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < 10; ++i) { GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS); }
            backend_wall.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count()/10);
        }
        printf("FRAMEWORK backend=%s kv=%d rows=%d layout=%s baseline_ggml_wall_us=%.3f\n", ggml_backend_name(backend),
            kv, rows, layout == 0 ? "linear" : "permuted", quantile(backend_wall, 0.5));
    } else {
        printf("CHECK kv=%d rows=%d top=%d layout=%d strided=%d: %zu candidates byte-exact against backend and std::sort, guards intact\n",
            kv, rows, top_count, layout, strided, std::size(names));
    }
    QSA_HIP_CHECK(hipFree(output));
    QSA_HIP_CHECK(hipStreamDestroy(stream));
    fflush(stdout);
}

int gpu_suite(const char * backend_name) {
    ggml_backend_load_all();
    ggml_backend_ptr backend(ggml_backend_init_by_name(backend_name, nullptr));
    GGML_ASSERT(backend);
    const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
    GGML_ASSERT(strcmp(ggml_backend_reg_name(reg), "ROCm") == 0);
    bool printed_device = false;
    int number = 0;
    for (int top : {1, 7, 16, 127, 255, 512, 1023}) {
        for (int layout = 0; layout < 4; ++layout) {
            const int rows = number%3 == 0 ? 1 : number%3 == 1 ? 3 : 8;
            run_case(backend.get(), 8192, rows, top, layout, number%2 != 0, false, printed_device);
            ++number;
        }
    }
    for (int layout : {0, 1}) {
        for (int rows : {1, 2, 3, 4, 5, 6, 7, 8, 32, 512, 1024}) {
            run_case(backend.get(), 235520, rows, 512, layout, false, true, printed_device);
        }
    }
    printf("QSA_SELECT candidate suite passed. Production selection and crossover remain at e7b431d.\n");
    return 0;
}

} // namespace qsa_test
