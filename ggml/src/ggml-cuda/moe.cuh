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
    if (packet.padding_bytes) {
        GGML_ASSERT(packet.padding);
        CUDA_CHECK(cudaMemsetAsync(packet.padding, 0, packet.padding_bytes, stream));
    }
    return true;
}
