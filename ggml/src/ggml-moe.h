#pragma once

// Process-local expert streaming contract. Not a graph serialization / RPC ABI.
#include "ggml.h"
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_backend;

// Native MMQ formats used by mixed expert models; FP4 formats with separate
// activation-scale semantics retain the ordinary path.
#ifdef __cplusplus
static constexpr bool ggml_moe_stream_type(enum ggml_type type) {
#else
static inline bool ggml_moe_stream_type(enum ggml_type type) {
#endif
    switch (type) {
        case GGML_TYPE_Q1_0: case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_S: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ4_XS:
            return true;
        default: return false;
    }
}

// Streaming eligibility follows the native MMQ weight format. MMB's stricter
// K % 256 requirement belongs to its own per-operation dispatch: the real
// Flash-Next down projection has K=640 and uses IQ4_NL / Q2_0 via MMQ.
static inline bool ggml_moe_stream_weight_supported(const struct ggml_tensor * w) {
    return ggml_moe_stream_type(w->type) && ggml_is_contiguous(w) && w->ne[3] == 1 &&
        w->ne[0] > 0 && w->ne[0] <= INT32_MAX && w->ne[0] % ggml_blck_size(w->type) == 0 &&
        w->ne[1] > 0 && w->ne[1] <= INT32_MAX && w->ne[1] % 128 == 0;
}

// Addresses are indexed by the model's logical expert ID, never by cache capacity.
// Only experts in this packet are active. Every selected expert must occur once.
// prepare queues uploads before its kernels, optionally on a fenced upload stream.
// Pointers remain valid until that stream has finished consuming the packet.
struct ggml_moe_packet {
    const uint64_t * addresses;
    const int32_t * experts;
    int32_t count;
    int32_t max_tokens;
    int32_t rows;
    // Reused staging may contain another format beyond the current payload.
    // Clear this read guard on the executing stream before consuming the packet.
    void * padding;
    size_t padding_bytes;
    // Optional backend events: uploads complete before reads; reuse follows reads.
    void * ready;
    void * consumed;
};

// Internal HIP service. Slot maps stay on device; only admission needs a host
// decision. Notifications also observe hits without blocking device execution.
typedef void (*ggml_moe_resolve_t)(void * userdata, int32_t layer, const int32_t * ids, int32_t count, bool miss);
struct ggml_moe_pipeline {
    void * context;
    bool device_lookup;
    void (*destroy)(void * context);
    void (*begin)(void * context);
    void (*flush)(void * context);
    void (*upload)(void * context, struct ggml_tensor * dst, const void * src, size_t offset, size_t bytes);
    void (*packet_begin)(void * context, int32_t lane);
    void (*packet_end)(void * context, int32_t lane, struct ggml_moe_packet * packet);
    void * (*lookup_job)(void * context, int32_t layer);
};
typedef bool (*ggml_moe_pipeline_init_t)(struct ggml_backend * backend, int32_t layers,
    ggml_moe_resolve_t resolve, void * userdata, struct ggml_moe_pipeline * out);

static inline struct ggml_tensor * ggml_moe_cache_lookup(struct ggml_context * ctx,
        struct ggml_tensor * map, struct ggml_tensor * ids, void * job) {
    GGML_ASSERT(map->type == GGML_TYPE_I32 && job);
    struct ggml_tensor * res = ggml_get_rows(ctx, map, ids);
    res->op = GGML_OP_MOE_CACHE_LOOKUP;
    memcpy(res->op_params, &job, sizeof(job));
    return res;
}

static inline void * ggml_moe_get_lookup_job(const struct ggml_tensor * t) {
    void * job;
    memcpy(&job, t->op_params, sizeof(job));
    return job;
}
typedef bool (*ggml_moe_prepare_t)(void * userdata, int32_t packet, struct ggml_moe_packet * out);
struct ggml_moe_stream {
    ggml_moe_prepare_t prepare;
    void * userdata;
    int32_t n_experts;
};

static inline struct ggml_moe_stream ggml_moe_get_stream(const struct ggml_tensor * t) {
    struct ggml_moe_stream res;
    // The first 16 bytes retain the usual matmul precision parameters.
    memcpy(&res, (const char *) t->op_params + 16, sizeof(res));
    return res;
}
static inline struct ggml_tensor * ggml_mul_mat_id_stream(
        struct ggml_context * ctx, struct ggml_tensor * bank, struct ggml_tensor * x,
        struct ggml_tensor * ids, struct ggml_tensor * ticket, struct ggml_moe_stream stream) {
    GGML_ASSERT(sizeof(stream) + 16 <= GGML_MAX_OP_PARAMS);
    GGML_ASSERT(stream.prepare && stream.n_experts > 0 && ticket && ggml_moe_stream_type(bank->type));
    struct ggml_tensor * res = ggml_mul_mat_id(ctx, bank, x, ids);
    res->op = GGML_OP_MUL_MAT_ID_STREAM;
    res->src[3] = ticket;
    memcpy((char *) res->op_params + 16, &stream, sizeof(stream));
    return res;
}

#ifdef __cplusplus
}
#endif
