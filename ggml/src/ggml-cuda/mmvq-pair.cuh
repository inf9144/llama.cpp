#pragma once

#include "common.cuh"
#include "vecdotq.cuh"

// Two independent projections, with the same Q8_1 input and the original per-row accumulation.
template <int ncols>
__launch_bounds__(32, 1)
static __global__ void mul_mat_vec_q6_k_pair(
        const void * first, const void * second, const block_q8_1 * input,
        float * first_dst, float * second_dst, int k, int first_rows, int second_rows, int input_stride) {
#if defined(RDNA3_0)
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;
    static_assert(warp_size == 32 && vdr == 2, "Match the RDNA3 Q6_K MMVQ reduction");
    static_assert(ncols >= 2 && ncols <= 4, "Verify projections only");
    const bool use_second = blockIdx.x >= uint32_t(first_rows);
    const int row = use_second ? blockIdx.x - first_rows : blockIdx.x;
    const void * weights = use_second ? second : first;
    float * dst = use_second ? second_dst : first_dst;
    const int rows = use_second ? second_rows : first_rows;
    const int blocks = k / QK_K;
    float sum[ncols] = {};
    ggml_cuda_pdl_sync();
    for (int b = threadIdx.x / (QI6_K/vdr); b < blocks; b += vdr*warp_size/QI6_K) {
        const int q = vdr*(threadIdx.x % (QI6_K/vdr));
#pragma unroll
        for (int col = 0; col < ncols; ++col) {
            sum[col] += vec_dot_q6_K_q8_1_vdr2(weights, input + col*input_stride + b*(QK_K/QK8_1), row*blocks + b, q);
        }
    }
#pragma unroll
    for (int col = 0; col < ncols; ++col) {
        sum[col] = warp_reduce_sum<warp_size>(sum[col]);
        if (threadIdx.x == 0) {
            dst[col*rows + row] = sum[col];
        }
    }
    ggml_cuda_pdl_lc();
#else
    GGML_UNUSED_VARS(first, second, input, first_dst, second_dst, k, first_rows, second_rows, input_stride);
    NO_DEVICE_CODE;
#endif
}
