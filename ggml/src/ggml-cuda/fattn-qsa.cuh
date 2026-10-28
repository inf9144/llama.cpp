#pragma once

#include "common.cuh"

#if defined(GGML_USE_HIP)
bool ggml_cuda_qsa_select_supported(const ggml_tensor * dst);
bool ggml_cuda_qsa_select_uses_radix(int device, const ggml_tensor * dst);
void ggml_cuda_qsa_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
bool ggml_cuda_flash_attn_sparse_supported(int device, const ggml_tensor * dst);
size_t ggml_cuda_flash_attn_sparse_get_alloc_size(const ggml_tensor * dst);
void ggml_cuda_flash_attn_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#endif
