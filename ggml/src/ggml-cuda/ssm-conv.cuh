#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

#if defined(GGML_USE_HIP)
void ggml_cuda_op_ssm_conv_rollback(ggml_backend_cuda_context & ctx, const ggml_tensor * input,
        const ggml_tensor * state_dst, size_t slot_stride);
#endif
