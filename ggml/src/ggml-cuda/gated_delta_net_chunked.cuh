#pragma once
#include "common.cuh"

// A rejected synchronous launch leaves the inputs unchanged; callers may use the sequential kernel.
template<typename Kernel, typename... Args>
static inline bool ggml_cuda_kernel_launch_try(Kernel kernel, const ggml_cuda_kernel_launch_params & p, Args&&... args) {
    kernel<<<p.block_nums, p.block_dims, p.shmem, p.stream>>>(std::forward<Args>(args)...);
    return cudaGetLastError() == cudaSuccess;
}

bool ggml_cuda_op_gated_delta_net_chunked(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_out, int64_t n_tokens);
bool ggml_cuda_op_gated_delta_net_chunked_bf16_gfx11(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * state_out, int64_t n_tokens);
