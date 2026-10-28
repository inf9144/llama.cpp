#include "../ggml-cuda/fattn-qsa.cuh"

#include "../ggml-cuda/fattn-tile.cuh"

struct qsa_select_params {
    const char * top;
    const char * score;
    const char * pools;
    const char * tail;
    int32_t * dst;
    int n_top, n_pool, n_sel, pool_size, n_kv, sort_size;
    size_t t0, t1, s0, s1, p0, p1, e0, e1;
};

__launch_bounds__(256)
static __global__ void qsa_select_kernel(const qsa_select_params p) {
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

struct qsa_attn_params {
    const char * q;
    const char * k;
    const char * v;
    const int32_t * indices;
    float * dst;
    float2 * meta;
    float scale;
    uint3 nq;
    int n_q, n_head, n_head_kv, n_kv, n_sel;
    size_t q1, q2, q3, k1, k2, k3, v1, v2, v3;
};

__launch_bounds__(128, 6)
static __global__ void qsa_attn_q8_kernel(const qsa_attn_params p) {
#if defined(RDNA3_0) && defined(FLASH_ATTN_AVAILABLE)
    constexpr int warp_size = 32;
    constexpr int nwarps = 4;
    constexpr int ncols2 = 4;
    constexpr int D = 256;
    constexpr int nbatch = 32;
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int seq = blockIdx.z / (p.n_head/ncols2);
    const int head = blockIdx.z*ncols2 - seq*p.n_head;
    const int kv_head = head / (p.n_head/p.n_head_kv);
    const int query = blockIdx.x;
    const float * q = (const float *) (p.q + seq*p.q3 + (head + warp)*p.q2 + query*p.q1);
    const int32_t * indices = p.indices + (int64_t(seq)*p.n_q + query)*p.n_sel;
    const fattn_kv_native_t kv = {
        p.k + seq*p.k3 + kv_head*p.k2, p.v + seq*p.v3 + kv_head*p.v2,
        int(p.k1), int(p.v1), FATTN_KV_NATIVE_Q8_0, FATTN_KV_NATIVE_Q8_0,
    };
    __shared__ half2 Q[ncols2*D/2];
    __shared__ half2 KV[nbatch*(D/2 + 4)];
    __shared__ half KQ[ncols2*nbatch];
    __align__(16) half2 acc[D/2/warp_size] = {{0.0f, 0.0f}};
    float max_val[1] = {-FLT_MAX/2.0f};
    float sum[1] = {0.0f};

    for (int i = lane; i < D/2; i += warp_size) {
        half2 val = make_half2(q[2*i]*p.scale, q[2*i + 1]*p.scale);
#if !defined(V_DOT2_F32_F16_AVAILABLE)
        val *= make_half2(0.25f, 0.25f);
#endif
        Q[warp*(D/2) + i] = val;
    }
    __syncthreads();

    const kq_derived_t derived = {nullptr, nullptr, nullptr};
    for (int i = blockIdx.y*nbatch; i < p.n_sel; i += gridDim.y*nbatch) {
        flash_attn_tile_iter<warp_size, nwarps, 1, ncols2, D, D, nbatch, D, false, true, GGML_TYPE_Q8_0, true>(
            Q, (const half2 *) nullptr, (const half2 *) nullptr, nullptr, p.nq, 0.0f, 1.0f, KQ, KV,
            0, 0, 0, derived, max_val, sum, acc, i, p.n_sel, query, kv, indices, p.n_kv);
    }
    sum[0] = warp_reduce_sum<warp_size>(sum[0]);
    const int64_t row = (int64_t(seq)*p.n_q + query)*p.n_head + head + warp;
    const int64_t part = row*gridDim.y + blockIdx.y;
    const float norm = gridDim.y == 1 ? (sum[0] == 0.0f ? 0.0f : 1.0f/sum[0]) : 1.0f;
    constexpr int cpy_ne = ggml_cuda_get_max_cpy_bytes()/sizeof(half2);
    constexpr int cpy_ne_D = cpy_ne/2 < D/2/warp_size ? cpy_ne/2 : D/2/warp_size;
    for (int i0 = 0; i0 < D/2; i0 += warp_size*cpy_ne_D) {
        for (int i1 = 0; i1 < cpy_ne_D; ++i1) {
            const float2 value = __half22float2(acc[i0/warp_size + i1]);
            const int d = 2*(i0 + lane*cpy_ne_D + i1);
            p.dst[part*D + d    ] = value.x*norm;
            p.dst[part*D + d + 1] = value.y*norm;
        }
    }
    if (gridDim.y > 1 && lane == 0) {
        p.meta[part] = make_float2(max_val[0], sum[0]);
    }
#else
    GGML_UNUSED(p);
    NO_DEVICE_CODE;
#endif
}

__launch_bounds__(256)
static __global__ void qsa_attn_combine(const float * parts, const float2 * meta, float * dst, const int n_parts) {
    const int64_t row = blockIdx.x;
    const int d = threadIdx.x;
    float max_val = meta[row*n_parts].x;
    for (int i = 1; i < n_parts; ++i) {
        max_val = fmaxf(max_val, meta[row*n_parts + i].x);
    }
    float num = 0.0f;
    float den = 0.0f;
    for (int i = 0; i < n_parts; ++i) {
        const float2 m = meta[row*n_parts + i];
        const float weight = expf(m.x - max_val);
        num += weight*parts[(row*n_parts + i)*256 + d];
        den += weight*m.y;
    }
    dst[row*256 + d] = den == 0.0f ? 0.0f : num/den;
}

bool ggml_cuda_qsa_select_supported(const ggml_tensor * dst) {
    return dst->type == GGML_TYPE_I32 && ggml_is_contiguous(dst) &&
        dst->ne[0] > 0 && dst->ne[0] <= 4096 && dst->ne[1] <= INT32_MAX && dst->src[1]->ne[0] <= INT32_MAX &&
        dst->src[0]->type == GGML_TYPE_I32 && dst->src[1]->type == GGML_TYPE_F32 &&
        dst->src[2]->type == GGML_TYPE_I32 && dst->src[3]->type == GGML_TYPE_I32;
}

void ggml_cuda_qsa_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    GGML_ASSERT(ggml_cuda_qsa_select_supported(dst));
    const ggml_tensor * t = dst->src[0];
    const ggml_tensor * s = dst->src[1];
    const ggml_tensor * b = dst->src[2];
    const ggml_tensor * e = dst->src[3];
    int sort_size = 256;
    while (sort_size < dst->ne[0]) {
        sort_size *= 2;
    }
    const qsa_select_params p = {
        (const char *) t->data, (const char *) s->data, (const char *) b->data, (const char *) e->data,
        (int32_t *) dst->data, int(t->ne[0]), int(s->ne[0]), int(dst->ne[0]), int(b->ne[0]),
        ggml_get_op_params_i32(dst, 0), sort_size,
        t->nb[0], t->nb[1], s->nb[0], s->nb[1], b->nb[0], b->nb[1], e->nb[0], e->nb[1],
    };
    const ggml_cuda_kernel_launch_params lp(dim3(dst->ne[1], 1, 1), dim3(256, 1, 1), (sort_size + 256)*sizeof(int32_t), ctx.stream());
    ggml_cuda_kernel_launch(qsa_select_kernel, lp, p);
    CUDA_CHECK(cudaGetLastError());
}

static int qsa_parallel_blocks(const ggml_tensor * dst) {
    // Keep the KV reduction identical for decode and verify, independent of query width.
    return dst->src[0]->ne[1] <= 8 ? 16 : 1;
}

size_t ggml_cuda_flash_attn_sparse_get_alloc_size(const ggml_tensor * dst) {
    const size_t size = ggml_nbytes(dst);
    const int n_parts = qsa_parallel_blocks(dst);
    if (n_parts == 1) {
        return size;
    }
    GGML_ASSERT(size <= SIZE_MAX - 127 && size <= SIZE_MAX/2);
    GGML_ASSERT(size_t(ggml_nrows(dst)) <= (SIZE_MAX - size)/sizeof(float2));
    const size_t part_size = size + ggml_nrows(dst)*sizeof(float2);
    GGML_ASSERT(part_size <= (SIZE_MAX - GGML_PAD(size, 128))/n_parts);
    return GGML_PAD(size, 128) + n_parts*part_size;
}

bool ggml_cuda_flash_attn_sparse_supported(int device, const ggml_tensor * dst) {
#if !defined(GGML_CUDA_NO_FA)
    const int cc = ggml_cuda_info().devices[device].cc;
    if (cc != GGML_CUDA_CC_RDNA3 && cc != GGML_CUDA_CC_RDNA3 + 1) {
        return false;
    }
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * idx = dst->src[3];
    if (q->ne[1] <= 0 || k->ne[1] <= 0 || k->ne[2] <= 0 ||
            q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_Q8_0 || v->type != GGML_TYPE_Q8_0 ||
            q->ne[0] != 256 || k->ne[0] != 256 || v->ne[0] != 256 ||
            q->ne[2] % k->ne[2] != 0 || (q->ne[2]/k->ne[2]) % 4 != 0 || k->ne[2] != v->ne[2] ||
            q->ne[3] != k->ne[3] || q->ne[3] != v->ne[3] || k->ne[1] != v->ne[1] ||
            idx->type != GGML_TYPE_I32 || !ggml_is_contiguous(idx) || idx->ne[0] <= 0 || idx->ne[0] > 4096 ||
            idx->ne[1] != q->ne[1] || idx->ne[2] != 1 || idx->ne[3] != q->ne[3]) {
        return false;
    }
    for (const ggml_tensor * t : {q, k, v}) {
        if (t->nb[0] != ggml_type_size(t->type)) {
            return false;
        }
        for (int i = 1; i < 4; ++i) {
            if (t->nb[i] > INT32_MAX) {
                return false;
            }
        }
    }
    return q->ne[1] <= 1024 && q->ne[2]*q->ne[3]/4 <= 65535;
#else
    GGML_UNUSED_VARS(device, dst);
    return false;
#endif
}

void ggml_cuda_flash_attn_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    GGML_ASSERT(ggml_cuda_flash_attn_sparse_supported(ctx.device, dst));
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * idx = dst->src[3];
    const int n_parts = qsa_parallel_blocks(dst);
    float * parts = n_parts == 1 ? (float *) dst->data : (float *) ((char *) dst->data + GGML_PAD(ggml_nbytes(dst), 128));
    float2 * meta = n_parts == 1 ? nullptr : (float2 *) (parts + n_parts*ggml_nelements(dst));
    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    const qsa_attn_params p = {
        (const char *) q->data, (const char *) k->data, (const char *) v->data, (const int32_t *) idx->data,
        parts, meta, scale, init_fastdiv_values(q->ne[1]), int(q->ne[1]), int(q->ne[2]), int(k->ne[2]), int(k->ne[1]), int(idx->ne[0]),
        q->nb[1], q->nb[2], q->nb[3], k->nb[1], k->nb[2], k->nb[3], v->nb[1], v->nb[2], v->nb[3],
    };
    const ggml_cuda_kernel_launch_params lp(dim3(q->ne[1], n_parts, q->ne[2]*q->ne[3]/4), dim3(32, 4, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(qsa_attn_q8_kernel, lp, p);
    CUDA_CHECK(cudaGetLastError());
    if (n_parts > 1) {
        const ggml_cuda_kernel_launch_params combine_lp(dim3(ggml_nrows(dst), 1, 1), dim3(256, 1, 1), 0, ctx.stream());
        ggml_cuda_kernel_launch(qsa_attn_combine, combine_lp, parts, meta, (float *) dst->data, n_parts);
        CUDA_CHECK(cudaGetLastError());
    }
}
