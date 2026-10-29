#pragma once

// Process-local expert streaming contract. Not a graph serialization / RPC ABI.
#include "ggml.h"
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

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

// Addresses are indexed by the model's logical expert ID, never by cache capacity.
// Only experts in this packet are active. Every selected expert must occur once.
// prepare queues uploads on the executing backend stream before its kernels run.
// Pointers remain valid until that stream has finished consuming the packet.
struct ggml_moe_packet {
    const uint64_t * addresses;
    const int32_t * experts;
    int32_t count;
    int32_t max_tokens;
    int32_t rows;
};
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
