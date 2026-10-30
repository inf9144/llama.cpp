#pragma once

#include "common.cuh"
#include "../ggml-moe.h"

// MMQ can load a full final K tile even when K is not a multiple of its tile
// width. Zero activation padding does not neutralize NaN weight scales. The
// guard must be re-cleared after every packet / weight-format reuse, ordered
// after its uploads and before its kernels, without a host synchronization.
static inline bool ggml_cuda_moe_prepare_packet(
        const ggml_moe_stream & source, int32_t index, ggml_moe_packet & packet, cudaStream_t stream) {
    if (!source.prepare(source.userdata, index, &packet)) return false;
    if (packet.ready) CUDA_CHECK(cudaStreamWaitEvent(stream, (cudaEvent_t) packet.ready, 0));
    if (packet.padding_bytes) {
        GGML_ASSERT(packet.padding);
        CUDA_CHECK(cudaMemsetAsync(packet.padding, 0, packet.padding_bytes, stream));
    }
    return true;
}

static inline void ggml_cuda_moe_consume_packet(const ggml_moe_packet & packet, cudaStream_t stream) {
    if (packet.consumed) CUDA_CHECK(cudaEventRecord((cudaEvent_t) packet.consumed, stream));
}

#if defined(GGML_USE_HIP)
void ggml_cuda_set_tensor_async_stream(ggml_backend_cuda_context & ctx, ggml_tensor * tensor,
    const void * data, size_t offset, size_t size, cudaStream_t stream);
bool ggml_hip_moe_pipeline_init(ggml_backend_t backend, int32_t layers,
    ggml_moe_resolve_t resolve, void * userdata, ggml_moe_pipeline * out);
void ggml_hip_moe_cache_lookup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
#endif
