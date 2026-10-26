#pragma once
#include "common.cuh"

bool ggml_cuda_mmb_supported_mm(int device, const ggml_tensor * w, const ggml_tensor * x, const ggml_tensor * dst);
bool ggml_cuda_mmb_supported_mmid(int device, const ggml_tensor * w, const ggml_tensor * x, const ggml_tensor * ids, const ggml_tensor * dst);
bool ggml_cuda_mmb_supported_glu(int device, const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * x, const ggml_tensor * ids, const ggml_tensor * dst);
void ggml_cuda_mul_mat_mmb(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x, ggml_tensor * dst);
void ggml_cuda_mul_mat_id_mmb(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x, const ggml_tensor * ids, ggml_tensor * dst);
void ggml_cuda_mul_mat_id_mmb_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * x, const ggml_tensor * ids, ggml_tensor * dst);
