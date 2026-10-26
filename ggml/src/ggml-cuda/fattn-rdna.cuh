#pragma once

#include "common.cuh"

// Share the native Q8_0 policy between scratch allocation and kernel launch.
static constexpr int GGML_CUDA_FA_Q8_CHUNK = 8;

enum fattn_kv_native_type : int {
    FATTN_KV_NATIVE_NONE = 0,
    FATTN_KV_NATIVE_Q8_0 = 1,
};

static constexpr int FATTN_KV_NATIVE_PER_OPERAND = -1;

static inline bool ggml_cuda_fattn_kv_native_supported(const ggml_tensor * t) {
#ifdef FAST_FP16_AVAILABLE
    return t != nullptr && t->type == GGML_TYPE_Q8_0 && t->ne[0] % GGML_CUDA_FA_Q8_CHUNK == 0 &&
        t->nb[0] == sizeof(block_q8_0);
#else
    GGML_UNUSED(t);
    return false;
#endif
}

static inline int ggml_cuda_fattn_kv_native_type(const ggml_tensor * t) {
    return ggml_cuda_fattn_kv_native_supported(t) ? FATTN_KV_NATIVE_Q8_0 : FATTN_KV_NATIVE_NONE;
}

template <ggml_type type_KV>
constexpr int ggml_cuda_fattn_native_type_from_kernel() {
    return type_KV == GGML_TYPE_Q8_0 ? FATTN_KV_NATIVE_Q8_0 : FATTN_KV_NATIVE_NONE;
}

static inline ggml_type ggml_cuda_fattn_tile_type(const ggml_tensor * K, const ggml_tensor * V, const int cc) {
    return fast_fp16_available(cc) && ggml_cuda_fattn_kv_native_supported(K) && ggml_cuda_fattn_kv_native_supported(V)
        ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
}
