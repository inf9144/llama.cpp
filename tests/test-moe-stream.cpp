#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-moe.h"
#include "../src/llama-moe-cache-policy.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

// The physical bank deliberately has fewer slots than the logical expert space.
// Both kernels read exactly the same quantized matrices. All activation values
// are exactly representable in BF16 and Q8, isolating address/routing correctness.
struct fixture {
    static constexpr int experts = 67, slots = 24, used = 4;
    int m = 128;
    int k = 256;
    ggml_backend_t backend;
    ggml_tensor * weights, * bank, * stage, * addresses, * active;
    std::vector<uint8_t> host;
    std::vector<uint8_t> poison;
    std::vector<int32_t> counts;
    std::vector<std::vector<int32_t>> packets;
    std::vector<std::vector<uint64_t>> maps;
    int stage_width;
    bool all_temporary;
    ggml_moe_pipeline pipeline = {};
    size_t stage_span = 0;

    static bool prepare(void * userdata, int32_t p, ggml_moe_packet * out) {
        return static_cast<fixture *>(userdata)->prepare(p, out);
    }
    bool prepare(int32_t p, ggml_moe_packet * out) {
        if (size_t(p) == packets.size()) return false;
        auto & map = maps[p];
        const auto & selected = packets[p];
        const int lane = pipeline.context ? p%2 : 0;
        auto upload = [&](ggml_tensor * dst, const void * src, size_t offset, size_t bytes) {
            if (pipeline.context) pipeline.upload(pipeline.context, dst, src, offset, bytes);
            else ggml_backend_tensor_set_async(backend, dst, src, offset, bytes);
        };
        if (pipeline.context) pipeline.packet_begin(pipeline.context, lane);
        std::fill(map.begin(), map.end(), 0);
        int max_tokens = 0, rows = 0;
        if (p > 0) {
            // Model reuse of a larger / differently quantized matrix. 0xff is
            // a NaN half scale in the native down formats, not neutral padding.
            upload(stage, poison.data(), lane*stage_span, poison.size());
        }
        for (size_t i = 0; i < selected.size(); ++i) {
            const int e = selected[i];
            if (p == 0) {
                map[e] = uintptr_t(bank->data) + size_t(slots - 1 - e/3)*bank->nb[2];
            } else {
                upload(stage, host.data() + size_t(e)*weights->nb[2], lane*stage_span + i*weights->nb[2], weights->nb[2]);
                map[e] = uintptr_t(stage->data) + lane*stage_span + i*weights->nb[2];
            }
            max_tokens = std::max(max_tokens, counts[e]); rows += counts[e];
        }
        upload(addresses, map.data(), lane*experts*sizeof(uint64_t), map.size()*sizeof(uint64_t));
        if (!selected.empty()) upload(active, selected.data(), lane*experts*sizeof(int32_t), selected.size()*sizeof(int32_t));
        *out = {(const uint64_t *) addresses->data + lane*experts, (const int32_t *) active->data + lane*experts, int32_t(selected.size()), max_tokens, rows,
                p > 0 ? (char *) stage->data + lane*stage_span + selected.size()*weights->nb[2] : nullptr,
                p > 0 ? weights->nb[2] : 0, nullptr, nullptr};
        if (pipeline.context) pipeline.packet_end(pipeline.context, lane, out);
        return true;
    }

    void run(ggml_type type, int tokens, bool broadcast, int packet_width, bool temporary, size_t seed, int columns = 256, int output_rows = 128) {
        k = columns; m = output_rows;
        const bool verify = tokens <= 8;
        stage_width = packet_width; all_temporary = temporary;
        ggml_context_ptr ctx(ggml_init({64*ggml_tensor_overhead() + 3*ggml_graph_overhead(), nullptr, true}));
        weights = ggml_new_tensor_3d(ctx.get(), type, k, m, experts);
        const int lanes = pipeline.context ? 2 : 1;
        stage_span = (stage_width + 1)*weights->nb[2];
        stage = temporary ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, lanes*stage_span) :
            ggml_new_tensor_3d(ctx.get(), type, k, m, lanes*(stage_width + 1));
        bank = ggml_new_tensor_3d(ctx.get(), type, k, m, temporary ? used : slots);
        if (temporary) {
            // Remote-owner execution uses only a typed view of bounded scratch.
            // No full working bank is allocated or passed to the scheduler.
            bank->view_src = stage;
        }
        addresses = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, lanes*experts);
        active = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, lanes*experts);
        auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, broadcast ? 1 : used, tokens);
        auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, used, tokens);
        auto * ticket = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
        auto * reference = ggml_mul_mat_id(ctx.get(), weights, x, ids);
        auto * actual = ggml_mul_mat_id_stream(ctx.get(), bank, x, ids, ticket, {&fixture::prepare, this, experts});
        ggml_tensor * decode = nullptr;
        if (verify) {
            for (int t = 0; t < tokens; ++t) {
                auto * xi = ggml_view_3d(ctx.get(), x, k, x->ne[1], 1, x->nb[1], x->nb[2], t*x->nb[2]);
                auto * ii = ggml_view_2d(ctx.get(), ids, used, 1, ids->nb[1], t*ids->nb[1]);
                auto * yi = ggml_mul_mat_id(ctx.get(), weights, xi, ii);
                decode = decode ? ggml_concat(ctx.get(), decode, yi, 2) : yi;
            }
        }
        GGML_ASSERT(ggml_moe_stream_weight_supported(weights) && ggml_moe_stream_weight_supported(bank));
        GGML_ASSERT(ggml_backend_supports_op(backend, actual));
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        GGML_ASSERT(buffer);
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_clear(buffer.get(), 0);
        if (temporary) GGML_ASSERT(bank->data == stage->data && bank->buffer == stage->buffer);
        poison.assign(stage_span, 0xff);

        std::mt19937 gen(seed);
        std::vector<float> wf(ggml_nelements(weights)), importance(k, 1.0f);
        for (auto & v : wf) v = (int(gen() % 255) - 127) / 128.0f;
        host.resize(ggml_nbytes(weights));
        ggml_quantize_chunk(type, wf.data(), host.data(), 0, m*experts, k, importance.data());
        ggml_backend_tensor_set(weights, host.data(), 0, host.size());
        for (int e = 0; e < experts; e += 3) {
            if (!temporary) ggml_backend_tensor_set(bank, host.data() + size_t(e)*weights->nb[2], size_t(slots - 1 - e/3)*bank->nb[2], bank->nb[2]);
        }
        std::vector<float> input(ggml_nelements(x));
        for (size_t i = 0; i < input.size(); ++i) input[i] = i % 32 == 0 ? 127.0f/16 : (int(gen() % 255) - 127)/16.0f;
        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
        std::vector<int32_t> routing(ggml_nelements(ids));
        counts.assign(experts, 0);
        for (int t = 0; t < tokens; ++t) for (int u = 0; u < used; ++u) {
            const int e = (t*37 + (used-u)*11) % experts;
            routing[t*used + u] = e; ++counts[e];
        }
        ggml_backend_tensor_set(ids, routing.data(), 0, ggml_nbytes(ids));
        packets.assign(1, {});
        std::vector<int32_t> misses;
        for (int e = experts-1; e >= 0; --e) {
            if (!counts[e]) continue;
            if (e % 3 == 0 && !all_temporary) packets[0].push_back(e);
            else misses.push_back(e);
        }
        for (size_t first = 0; first < misses.size(); first += stage_width) {
            packets.emplace_back(misses.begin() + first, misses.begin() + std::min(first + stage_width, misses.size()));
        }
        maps.assign(packets.size(), std::vector<uint64_t>(experts));
        for (auto * out : {reference, actual, decode}) {
            if (!out) continue;
            ggml_cgraph * graph = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(graph, out);
            GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_synchronize(backend);
        }
        // Check the final partial packet as well as kernel finiteness. The CPU
        // oracle checks the same contract even though it reads only logical K.
        std::vector<uint8_t> guard(weights->nb[2], 0xff);
        const int last_lane = pipeline.context ? (packets.size() - 1)%2 : 0;
        ggml_backend_tensor_get(stage, guard.data(), last_lane*stage_span + packets.back().size()*weights->nb[2], guard.size());
        GGML_ASSERT(std::all_of(guard.begin(), guard.end(), [](uint8_t value) { return value == 0; }));
        std::vector<float> a(ggml_nelements(actual)), r(a.size());
        ggml_backend_tensor_get(actual, a.data(), 0, ggml_nbytes(actual));
        ggml_backend_tensor_get(reference, r.data(), 0, ggml_nbytes(reference));
        double error = 0, energy = 0, max_error = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            GGML_ASSERT(std::isfinite(a[i]) && std::isfinite(r[i]));
            const double d = double(a[i]) - r[i]; error += d*d; energy += double(r[i])*r[i]; max_error = std::max(max_error, std::abs(d));
        }
        const double nmse = error / std::max(energy, 1e-30);
        printf("%s %s K=%d tokens=%d broadcast=%d packet=%d all_stage=%d packets=%zu NMSE=%.9g max_abs=%.9g\n",
            ggml_backend_name(backend), ggml_type_name(type), k, tokens, broadcast, stage_width, all_temporary, packets.size(), nmse, max_error);
        fflush(stdout);
        if (verify) {
            // Neither GPU result is an oracle: decode all quantized weights on
            // CPU, then accumulate each routed dot product in double precision.
            std::vector<float> d(a.size()), row_weights(k);
            ggml_backend_tensor_get(decode, d.data(), 0, ggml_nbytes(decode));
            double error_verify = 0, error_stream = 0, error_decode = 0, error_consistency = 0, oracle_energy = 0;
            for (int t = 0; t < tokens; ++t) for (int u = 0; u < used; ++u) for (int row = 0; row < m; ++row) {
                const size_t offset = size_t(routing[t*used + u])*weights->nb[2] + row*weights->nb[1];
                ggml_get_type_traits(type)->to_float(host.data() + offset, row_weights.data(), k);
                const auto * values = input.data() + (t*x->ne[1] + (broadcast ? 0 : u))*k;
                double expected = 0;
                for (int col = 0; col < k; ++col) expected += double(row_weights[col])*values[col];
                const int i = (t*used + u)*m + row;
                const double v = double(r[i]) - expected, s = double(a[i]) - expected, dc = double(d[i]) - expected;
                const double consistent = double(r[i]) - d[i];
                GGML_ASSERT(std::isfinite(d[i]));
                error_verify += v*v; error_stream += s*s; error_decode += dc*dc; error_consistency += consistent*consistent;
                oracle_energy += expected*expected;
            }
            const double norm = std::max(oracle_energy, 1e-30);
            printf("%s %s verify oracle: verify=%.9g stream=%.9g decode=%.9g decode_vs_verify=%.9g (NMSE)\n",
                ggml_backend_name(backend), ggml_type_name(type), error_verify/norm, error_stream/norm, error_decode/norm, error_consistency/norm);
            fflush(stdout);
            GGML_ASSERT(error_verify/norm < 1e-10 && error_stream/norm < 1e-10 && error_decode/norm < 1e-10 && error_consistency/norm < 1e-10);
        }
        GGML_ASSERT(nmse < 1e-10);
    }
};

struct lookup_fixture {
    static constexpr int experts = 13, slots = 8, k = 256, m = 128, tokens = 8, used = 2;
    ggml_backend_t backend;
    ggml_moe_pipeline pipeline = {};
    ggml_tensor * map;
    ggml_tensor * bank;
    moe_cache_policy policy;
    std::vector<int32_t> maps = std::vector<int32_t>(2*experts, -1);
    std::vector<uint8_t> weights;
    uint64_t clock = 0;
    int hits = 0, misses = 0, cross_layer_evictions = 0;

    static void resolve(void * userdata, int32_t il, const int32_t * ids, int32_t count, bool miss) {
        auto & f = *(lookup_fixture *) userdata;
        std::vector<moe_cache_policy::fill> fills;
        size_t n_hit;
        const auto before = f.maps;
        GGML_ASSERT(f.policy.plan(il, ids, count, fills, n_hit, f.clock, true, f.clock, miss));
        if (!miss) { ++f.hits; return; }
        ++f.misses;
        for (int e = 0; e < experts; ++e) {
            const int index = (1 - il)*experts + e;
            f.cross_layer_evictions += before[index] >= 0 && f.maps[index] < 0;
        }
        for (auto fill : fills) {
            f.pipeline.upload(f.pipeline.context, f.bank,
                f.weights.data() + (il*experts + fill.expert)*f.bank->nb[2], fill.slot*f.bank->nb[2], f.bank->nb[2]);
        }
        f.pipeline.upload(f.pipeline.context, f.map, f.maps.data(), 0, ggml_nbytes(f.map));
    }

    void run(ggml_moe_pipeline_init_t init) {
        if (!init(backend, 2, &resolve, this, &pipeline)) return;
        if (!pipeline.device_lookup) {
            printf("%s device lookup tests SKIPPED: stream-signal graph support unavailable\n", ggml_backend_name(backend));
            pipeline.destroy(pipeline.context);
            return;
        }
        ggml_context_ptr ctx(ggml_init({64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true}));
        map = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 2*experts);
        bank = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q8_0, k, m, slots + 1);
        auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, used, tokens);
        ggml_tensor * ids[2], * out[2];
        auto * graph = ggml_new_graph(ctx.get());
        for (int il = 0; il < 2; ++il) {
            ids[il] = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, used*tokens);
            auto * view = ggml_view_2d(ctx.get(), map, 1, experts, sizeof(int32_t), il*experts*sizeof(int32_t));
            auto * selected = ggml_moe_cache_lookup(ctx.get(), view, ids[il], pipeline.lookup_job(pipeline.context, il));
            out[il] = ggml_mul_mat_id(ctx.get(), bank, x, ggml_reshape_2d(ctx.get(), selected, used, tokens));
            ggml_set_output(out[il]);
            ggml_build_forward_expand(graph, out[il]);
        }
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        GGML_ASSERT(buffer);
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_buffer_clear(buffer.get(), 0);
        ggml_backend_tensor_set(map, maps.data(), 0, ggml_nbytes(map));
        policy.init(2, experts, slots);
        for (int il = 0; il < 2; ++il) policy.slot_map[il] = maps.data() + il*experts;
        std::vector<float> source(2*experts*k*m), input(k*used*tokens);
        for (size_t i = 0; i < source.size(); ++i) source[i] = (int(i%251) - 125)/128.0f;
        for (size_t i = 0; i < input.size(); ++i) input[i] = i%32 == 0 ? 127.0f/16 : (int(i%127) - 63)/16.0f;
        weights.resize(2*experts*bank->nb[2]);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, source.data(), weights.data(), 0, 2*experts*m, k, nullptr);
        ggml_get_type_traits(GGML_TYPE_Q8_0)->to_float(weights.data(), source.data(), source.size());
        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
        double worst = 0;
        for (int step = 0; step < 12; ++step) {
            clock = (step + 1)*32;
            std::vector<int32_t> routing[2];
            for (int il = 0; il < 2; ++il) {
                routing[il].resize(used*tokens);
                const int period = il == 0 && step >= 4 ? 3 : 2;
                for (int t = 0; t < tokens; ++t) for (int u = 0; u < used; ++u) routing[il][t*used + u] = (3*(step/2) + 2*(t%period) + u)%experts;
                ggml_backend_tensor_set(ids[il], routing[il].data(), 0, ggml_nbytes(ids[il]));
            }
            pipeline.begin(pipeline.context);
            GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            pipeline.flush(pipeline.context);
            for (int il = 0; il < 2; ++il) {
                std::vector<float> actual(m*used*tokens);
                ggml_backend_tensor_get(out[il], actual.data(), 0, ggml_nbytes(out[il]));
                double error = 0, energy = 0;
                for (int t = 0; t < tokens; ++t) for (int u = 0; u < used; ++u) for (int row = 0; row < m; ++row) {
                    double expected = 0;
                    const int e = il*experts + routing[il][t*used + u];
                    for (int col = 0; col < k; ++col) expected += double(source[(e*m + row)*k + col])*input[(t*used + u)*k + col];
                    const double value = actual[(t*used + u)*m + row];
                    GGML_ASSERT(std::isfinite(value));
                    error += (value - expected)*(value - expected); energy += expected*expected;
                }
                worst = std::max(worst, error/std::max(energy, 1e-30));
            }
        }
        GGML_ASSERT(hits > 0 && misses > 0 && cross_layer_evictions > 0 && worst < 1e-10);
        printf("%s device lookup: hits=%d misses=%d cross-layer-evictions=%d, cold/admission and 12 graph calls with capture/replay, NMSE=%g\n",
            ggml_backend_name(backend), hits, misses, cross_layer_evictions, worst);
        pipeline.destroy(pipeline.context);
    }
};

// Reproduce the poisoned-tail failure without requiring a GPU: a tiled read
// dequantizes the next block, then multiplies it by zero activation padding.
static void test_quantized_padding() {
    ggml_context_ptr init(ggml_init({ggml_tensor_overhead(), nullptr, true}));
    GGML_ASSERT(init);
    constexpr int logical_k = 640, tiled_k = 768;
    for (auto type : {GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL, GGML_TYPE_Q2_0}) {
        const size_t payload = ggml_row_size(type, logical_k), padded = ggml_row_size(type, tiled_k);
        std::vector<float> weights(logical_k, 0.25f), dequantized(tiled_k);
        std::vector<uint8_t> bytes(padded, 0xff);
        ggml_quantize_chunk(type, weights.data(), bytes.data(), 0, 1, logical_k, nullptr);
        const auto to_float = ggml_get_type_traits(type)->to_float;
        to_float(bytes.data(), dequantized.data(), tiled_k);
        volatile float zero = 0.0f;
        float tail = 0;
        for (int i = logical_k; i < tiled_k; ++i) tail += zero*dequantized[i];
        GGML_ASSERT(std::isnan(tail));
        memset(bytes.data() + payload, 0, padded - payload);
        to_float(bytes.data(), dequantized.data(), tiled_k);
        tail = 0;
        for (int i = logical_k; i < tiled_k; ++i) tail += zero*dequantized[i];
        GGML_ASSERT(std::isfinite(tail) && tail == 0);
    }
    printf("Quantized K=640 padding: poisoned scales reproduce NaNs; zero guards restore finite tails in 3 formats\n");
}

int main(int argc, char ** argv) {
    const char * filter = argc == 3 && strcmp(argv[1], "--backend") == 0 ? argv[2] : nullptr;
    GGML_ASSERT(argc == 1 || filter);
    ggml_backend_load_all();
    test_quantized_padding();
    int tested = 0, gpu = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        if (filter && strcmp(filter, ggml_backend_dev_name(dev))) continue;
        auto reg = ggml_backend_dev_backend_reg(dev);
        using supports_fn = bool (*)(ggml_backend_dev_t);
        auto supports = reg ? (supports_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hip_moe_cache_supports_prefill") : nullptr;
        const bool hip = supports && supports(dev);
        if (!hip && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) continue;
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr)); GGML_ASSERT(backend);
        if (ggml_backend_is_cpu(backend.get())) ggml_backend_cpu_set_n_threads(backend.get(), 2);
        fixture f; f.backend = backend.get();
        auto init = reg ? (ggml_moe_pipeline_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hip_moe_pipeline_init") : nullptr;
        if (hip && init) init(backend.get(), 1, [](void *, int32_t, const int32_t *, int32_t, bool) {}, nullptr, &f.pipeline);
        for (auto type : {GGML_TYPE_Q8_0, GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_Q2_0}) {
            f.run(type, 64, true, 8, false, 42); // MMQ, repeated packets and tail
            f.run(type, 512, true, 32, false, 43); // MMB prefill / up
            f.run(type, 512, false, 16, true, 44); // MMB down, empty resident packet
        }
        // Real model FFN width: a partial 256-column iteration is valid MMQ
        // input. MMB must decline it without disabling the layer's stream plan.
        for (auto type : {GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL, GGML_TYPE_Q2_0}) {
            f.run(type, 64, false, 8, false, 45, 640);
            f.run(type, 512, false, 32, true, 46, 640);
        }
        if (hip) {
            for (auto type : {GGML_TYPE_Q8_0, GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_Q2_0}) {
                for (int width : {5, 8}) f.run(type, width, true, 8, false, 47, 2560, 640);
            }
            for (auto type : {GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL, GGML_TYPE_Q2_0}) {
                for (int width : {5, 8}) f.run(type, width, false, 8, true, 48, 640, 2560);
            }
        }
        if (f.pipeline.context) f.pipeline.destroy(f.pipeline.context);
        if (hip && init) { lookup_fixture lookup; lookup.backend = backend.get(); lookup.run(init); }
        ++tested; gpu += hip;
    }
    if (!gpu) printf("RDNA3 HIP devices unavailable: GPU kernel tests SKIPPED\n");
    GGML_ASSERT(tested > 0);
    ggml_quantize_free();
}
