#pragma once

#include "common.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

struct mmvq_moe_group {
    int expert;
    int count;
    int rows[4];
};

static __global__ void mmvq_moe_group_ids(const int32_t * ids, int used, int stride, mmvq_moe_group * groups) {
    __shared__ int selected[64];
    __shared__ int leaders[64];
    const int i = threadIdx.x;
    const int n = 4*used;
    selected[i] = i < n ? ids[(i/used)*stride + i%used] : -1;
    __syncthreads();

    mmvq_moe_group group = {};
    if (i < n) {
        group.expert = selected[i];
        int ordinal = 0;
        for (int j = (i/used)*used; j < i; ++j) {
            ordinal += selected[j] == group.expert;
        }
        for (int token = 0; token < 4; ++token) {
            int occurrence = 0;
            for (int rank = 0; rank < used; ++rank) {
                const int row = token*used + rank;
                if (selected[row] == group.expert && occurrence++ == ordinal) {
                    group.rows[group.count++] = row;
                    break;
                }
            }
        }
    }
    leaders[i] = i < n && group.rows[0] == i;
    __syncthreads();
    int offset = 0;
    for (int j = 0; j < i; ++j) {
        offset += leaders[j];
    }
    if (leaders[i]) {
        groups[offset] = group;
    }
    if (i == 63) {
        groups[n].count = offset + leaders[i];
    }
}

struct mmvq_moe_decoded {
    int2 qs[4];
    float d;
    int ls0;
    int ls1;
};

template <ggml_type type>
static __device__ __forceinline__ mmvq_moe_decoded mmvq_moe_decode(const void * weights, int block, int iqs) {
    mmvq_moe_decoded v = {};
    if constexpr (type == GGML_TYPE_IQ2_S) {
        const auto * b = (const block_iq2_s *) weights + block;
        const int qs = get_int_b2(b->qs, iqs/2);
        const int signs = get_int_b2(b->qs, QK_K/32 + iqs/2);
        const auto * q = (const uint8_t *) &qs;
        const auto * s = (const uint8_t *) &signs;
        const int qh = b->qh[iqs/2];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const auto * grid = (const int *) (iq2s_grid + (q[j] | ((qh << (8 - 2*j)) & 0x300)));
            v.qs[j] = make_int2(apply_ksigns(grid[0], s[j], 0), apply_ksigns(grid[1], s[j], 4));
        }
        v.d = __half2float(b->d);
        v.ls0 = b->scales[iqs/2] & 15;
        v.ls1 = b->scales[iqs/2] >> 4;
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        const auto * b = (const block_iq3_xxs *) weights + block;
        const int2 packed = make_int2(get_int_b2(b->qs, iqs), get_int_b2(b->qs, iqs + 1));
        const auto * q = (const uint8_t *) &packed;
        const uint32_t aux = get_int_b2(b->qs, QK_K/16 + iqs/2);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t signs = unpack_ksigns(aux >> (7*j));
            v.qs[j] = make_int2(apply_ksigns(iq3xxs_grid[q[2*j]], signs, 0), apply_ksigns(iq3xxs_grid[q[2*j + 1]], signs, 4));
        }
        v.d = __half2float(b->d);
        v.ls0 = 2*(aux >> 28) + 1;
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        const auto * b = (const block_iq3_s *) weights + block;
        const int2 packed = make_int2(get_int_b2(b->qs, iqs), get_int_b2(b->qs, iqs + 1));
        const int signs = get_int_b2(b->signs, iqs/2);
        const auto * q = (const uint8_t *) &packed;
        const auto * s = (const uint8_t *) &signs;
        const int qh = b->qh[iqs/2];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int l = iq3s_grid[q[2*j] | ((qh << (8 - 2*j)) & 0x100)];
            const int h = iq3s_grid[q[2*j + 1] | ((qh << (7 - 2*j)) & 0x100)];
            v.qs[j] = make_int2(apply_ksigns(l, s[j], 0), apply_ksigns(h, s[j], 4));
        }
        v.d = __half2float(b->d);
        v.ls0 = 1 + 2*((b->scales[iqs/4] >> ((iqs << 1) & 4)) & 15);
    } else if constexpr (type == GGML_TYPE_IQ4_NL) {
        const auto * b = (const block_iq4_nl *) weights + block;
#pragma unroll
        for (int j = 0; j < VDR_IQ4_NL_Q8_1_MMVQ; ++j) {
            v.qs[j] = get_int_from_table_16(get_int_b2(b->qs, iqs + j), kvalues_iq4nl);
        }
        v.d = __half2float(b->d);
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        const auto * b = (const block_iq4_xs *) weights + block;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v.qs[j] = get_int_from_table_16(get_int_b4(b->qs, iqs + j), kvalues_iq4nl);
        }
        v.d = __half2float(b->d);
        v.ls0 = ((b->scales_l[iqs/8] >> (iqs & 4)) & 15) | (((b->scales_h >> (iqs/2)) & 3) << 4);
        v.ls0 -= 32;
    } else {
        static_assert(type == GGML_TYPE_Q2_0, "unsupported grouped MoE quantization");
        const auto * b = (const block_q2_0 *) weights + block;
        const auto * qs = (const int16_t *) b->qs + iqs*4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const uint32_t lo = (q & 3) | ((q & 12) << 6) | ((q & 48) << 12) | ((q & 192) << 18);
            const uint32_t h = q >> 8;
            const uint32_t hi = (h & 3) | ((h & 12) << 6) | ((h & 48) << 12) | ((h & 192) << 18);
            v.qs[j] = make_int2(ggml_hip_perm(0x020100FF, 0x020100FF, lo), ggml_hip_perm(0x020100FF, 0x020100FF, hi));
        }
        v.d = b->d;
    }
    return v;
}

template <ggml_type type>
static __device__ __forceinline__ float mmvq_moe_dot(const mmvq_moe_decoded & v, const block_q8_1 * y, int iqs) {
    int sum = 0;
    int sum1 = 0;
    if constexpr (type == GGML_TYPE_IQ4_NL) {
#pragma unroll
        for (int j = 0; j < VDR_IQ4_NL_Q8_1_MMVQ; ++j) {
            sum = ggml_cuda_dp4a(v.qs[j].x, get_int_b4(y->qs, iqs + j), sum);
            sum = ggml_cuda_dp4a(v.qs[j].y, get_int_b4(y->qs, iqs + j + 4), sum);
        }
    } else {
        y += type == GGML_TYPE_Q2_0 ? iqs : type == GGML_TYPE_IQ4_XS ? iqs/4 : iqs/2;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int lo = type == GGML_TYPE_IQ4_XS ? j : 2*j;
            const int hi = type == GGML_TYPE_IQ4_XS ? j + 4 : 2*j + 1;
            if constexpr (type == GGML_TYPE_IQ2_S) {
                if (j >= 2) {
                    sum1 = ggml_cuda_dp4a(v.qs[j].x, get_int_b4(y->qs, lo), sum1);
                    sum1 = ggml_cuda_dp4a(v.qs[j].y, get_int_b4(y->qs, hi), sum1);
                    continue;
                }
            }
            sum = ggml_cuda_dp4a(v.qs[j].x, get_int_b4(y->qs, lo), sum);
            sum = ggml_cuda_dp4a(v.qs[j].y, get_int_b4(y->qs, hi), sum);
        }
    }
    if constexpr (type == GGML_TYPE_IQ2_S) {
        sum = ggml_cuda_mul_small(sum, 2*v.ls0 + 1) + ggml_cuda_mul_small(sum1, 2*v.ls1 + 1);
    } else if constexpr (type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ4_XS) {
        sum = ggml_cuda_mul_small(sum, v.ls0);
    }
    float d = v.d * __low2float(y->ds);
    if constexpr (type == GGML_TYPE_IQ2_S) {
        d *= 0.125f;
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        d *= 0.25f;
    }
    return d * sum;
}

template <ggml_type type>
static constexpr __host__ __device__ int mmvq_moe_group_vdr() {
    return type == GGML_TYPE_Q2_0 ? VDR_Q2_0_Q8_1_MMVQ : type == GGML_TYPE_IQ4_XS ? VDR_IQ4_XS_Q8_1_MMVQ : 2;
}

template <ggml_type type, bool fused>
__launch_bounds__(128, 1)
static __global__ void mul_mat_vec_q_moe_grouped(
        const void * weights, const block_q8_1 * input, const mmvq_moe_group * groups,
        ggml_cuda_mm_fusion_args_device fusion, float * dst, int k, int m, int used, int input_channels,
        int weight_row, int weight_expert, int input_token, int input_channel, int output_token, int output_channel) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = mmvq_moe_group_vdr<type>();
    const int count = groups[4*used].count;
    const bool shared = fused && fusion.shared_up && blockIdx.y == uint32_t(count);
    const int row = 8*blockIdx.x + 2*threadIdx.y;
    if (row >= m || (blockIdx.y >= uint32_t(count) && !shared)) {
        return;
    }
    mmvq_moe_group group;
    const void * gate = fusion.gate;
    if (shared) {
        group.expert = 0;
        group.count = 4;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            group.rows[t] = t*used;
        }
        weights = fusion.shared_up;
        gate = fusion.shared_gate;
        dst = fusion.shared_dst;
        output_token = fusion.shared_stride_col_dst;
        output_channel = 0;
    } else {
        group = groups[blockIdx.y];
    }
    const block_q8_1 * y[4];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        y[t] = input;
        if (t < group.count) {
            const int token = group.rows[t]/used;
            const int rank = group.rows[t]%used;
            y[t] += token*input_token + (rank%input_channels)*input_channel;
        }
    }
    float sum[4][2] = {};
    float sum_gate[4][2] = {};
    const int base = group.expert*weight_expert + row*weight_row;
    for (int block = threadIdx.x/(qi/vdr); block < k/qk; block += vdr*warp_size/qi) {
        const int iqs = vdr*(threadIdx.x%(qi/vdr));
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (row + r >= m) {
                continue;
            }
            // Decode each weight chunk once, then keep the original dot and reduction order per token.
            const auto values = mmvq_moe_decode<type>(weights, base + r*weight_row + block, iqs);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                if (t < group.count) {
                    sum[t][r] += mmvq_moe_dot<type>(values, y[t] + block*(qk/QK8_1), iqs);
                }
            }
            if constexpr (fused) {
                const auto values_gate = mmvq_moe_decode<type>(gate, base + r*weight_row + block, iqs);
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    if (t < group.count) {
                        sum_gate[t][r] += mmvq_moe_dot<type>(values_gate, y[t] + block*(qk/QK8_1), iqs);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        if (t < group.count) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                sum[t][r] = warp_reduce_sum<warp_size>(sum[t][r]);
                if constexpr (fused) {
                    sum_gate[t][r] = warp_reduce_sum<warp_size>(sum_gate[t][r]);
                }
            }
            if (threadIdx.x < 2 && row + int(threadIdx.x) < m) {
                float value = sum[t][threadIdx.x];
                if constexpr (fused) {
                    const float g = sum_gate[t][threadIdx.x];
                    switch (fusion.glu_op) {
                        case GGML_GLU_OP_SWIGLU:       value *= ggml_cuda_op_silu_single(g); break;
                        case GGML_GLU_OP_GEGLU:        value *= ggml_cuda_op_gelu_single(g); break;
                        case GGML_GLU_OP_SWIGLU_OAI:   value = ggml_cuda_op_swiglu_oai_single(g, value); break;
                        case GGML_GLU_OP_SWIGLU_CLAMP: value = ggml_cuda_op_swiglu_clamp_single(g, value, fusion.glu_limit); break;
                        default:                      value *= g; break;
                    }
                }
                const int token = group.rows[t]/used;
                const int rank = group.rows[t]%used;
                dst[token*output_token + rank*output_channel + row + threadIdx.x] = value;
            }
        }
    }
}

template <ggml_type type>
static void mul_mat_vec_q_moe_grouped_launch(
        const void * weights, const block_q8_1 * input, const mmvq_moe_group * groups,
        ggml_cuda_mm_fusion_args_device fusion, float * dst, int k, int m, int used, int input_channels,
        int weight_row, int weight_expert, int input_token, int input_channel, int output_token, int output_channel,
        int warp_size, cudaStream_t stream) {
    const dim3 blocks((m + 7)/8, 4*used + (fusion.shared_up != nullptr));
    const dim3 threads(warp_size, 4);
    if (fusion.gate) {
        mul_mat_vec_q_moe_grouped<type, true><<<blocks, threads, 0, stream>>>(weights, input, groups, fusion, dst, k, m, used, input_channels,
                weight_row, weight_expert, input_token, input_channel, output_token, output_channel);
    } else {
        mul_mat_vec_q_moe_grouped<type, false><<<blocks, threads, 0, stream>>>(weights, input, groups, fusion, dst, k, m, used, input_channels,
                weight_row, weight_expert, input_token, input_channel, output_token, output_channel);
    }
}
