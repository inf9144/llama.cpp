#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-moe.h"

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
    static constexpr int experts = 67, slots = 24, k = 256, m = 128, used = 4;
    ggml_backend_t backend;
    ggml_tensor * weights, * bank, * stage, * addresses, * active;
    std::vector<uint8_t> host;
    std::vector<int32_t> counts;
    std::vector<std::vector<int32_t>> packets;
    std::vector<std::vector<uint64_t>> maps;
    int stage_width;
    bool all_temporary;

    static bool prepare(void * userdata, int32_t p, ggml_moe_packet * out) {
        return static_cast<fixture *>(userdata)->prepare(p, out);
    }
    bool prepare(int32_t p, ggml_moe_packet * out) {
        if (size_t(p) == packets.size()) return false;
        auto & map = maps[p];
        const auto & selected = packets[p];
        std::fill(map.begin(), map.end(), 0);
        int max_tokens = 0, rows = 0;
        for (size_t i = 0; i < selected.size(); ++i) {
            const int e = selected[i];
            if (p == 0) {
                map[e] = uintptr_t(bank->data) + size_t(slots - 1 - e/3)*bank->nb[2];
            } else {
                ggml_backend_tensor_set_async(backend, stage, host.data() + size_t(e)*weights->nb[2], i*weights->nb[2], weights->nb[2]);
                map[e] = uintptr_t(stage->data) + i*weights->nb[2];
            }
            max_tokens = std::max(max_tokens, counts[e]); rows += counts[e];
        }
        ggml_backend_tensor_set_async(backend, addresses, map.data(), 0, map.size()*sizeof(uint64_t));
        if (!selected.empty()) ggml_backend_tensor_set_async(backend, active, selected.data(), 0, selected.size()*sizeof(int32_t));
        *out = {(const uint64_t *) addresses->data, (const int32_t *) active->data, int32_t(selected.size()), max_tokens, rows};
        return true;
    }

    void run(ggml_type type, int tokens, bool broadcast, int packet_width, bool temporary, size_t seed) {
        stage_width = packet_width; all_temporary = temporary;
        ggml_context_ptr ctx(ggml_init({64*ggml_tensor_overhead() + 2*ggml_graph_overhead(), nullptr, true}));
        weights = ggml_new_tensor_3d(ctx.get(), type, k, m, experts);
        bank = ggml_new_tensor_3d(ctx.get(), type, k, m, slots);
        stage = ggml_new_tensor_3d(ctx.get(), type, k, m, stage_width + 1);
        addresses = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, experts);
        active = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, experts);
        auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, broadcast ? 1 : used, tokens);
        auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, used, tokens);
        auto * ticket = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
        auto * reference = ggml_mul_mat_id(ctx.get(), weights, x, ids);
        auto * actual = ggml_mul_mat_id_stream(ctx.get(), bank, x, ids, ticket, {&fixture::prepare, this, experts});
        GGML_ASSERT(ggml_backend_supports_op(backend, actual));
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        GGML_ASSERT(buffer);
        ggml_backend_buffer_clear(buffer.get(), 0);

        std::mt19937 gen(seed);
        std::vector<float> wf(ggml_nelements(weights)), importance(k, 1.0f);
        for (auto & v : wf) v = (int(gen() % 255) - 127) / 128.0f;
        host.resize(ggml_nbytes(weights));
        ggml_quantize_chunk(type, wf.data(), host.data(), 0, m*experts, k, importance.data());
        ggml_backend_tensor_set(weights, host.data(), 0, host.size());
        for (int e = 0; e < experts; e += 3) {
            ggml_backend_tensor_set(bank, host.data() + size_t(e)*weights->nb[2], size_t(slots - 1 - e/3)*bank->nb[2], bank->nb[2]);
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
        for (auto * out : {reference, actual}) {
            ggml_cgraph * graph = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(graph, out);
            GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
            ggml_backend_synchronize(backend);
        }
        std::vector<float> a(ggml_nelements(actual)), r(a.size());
        ggml_backend_tensor_get(actual, a.data(), 0, ggml_nbytes(actual));
        ggml_backend_tensor_get(reference, r.data(), 0, ggml_nbytes(reference));
        double error = 0, energy = 0, max_error = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            GGML_ASSERT(std::isfinite(a[i]) && std::isfinite(r[i]));
            const double d = double(a[i]) - r[i]; error += d*d; energy += double(r[i])*r[i]; max_error = std::max(max_error, std::abs(d));
        }
        const double nmse = error / std::max(energy, 1e-30);
        printf("%s %s tokens=%d broadcast=%d packet=%d all_stage=%d packets=%zu NMSE=%.9g max_abs=%.9g\n",
            ggml_backend_name(backend), ggml_type_name(type), tokens, broadcast, stage_width, all_temporary, packets.size(), nmse, max_error);
        GGML_ASSERT(nmse < 1e-10);
    }
};

int main(int argc, char ** argv) {
    const char * filter = argc == 3 && strcmp(argv[1], "--backend") == 0 ? argv[2] : nullptr;
    GGML_ASSERT(argc == 1 || filter);
    ggml_backend_load_all();
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
        for (auto type : {GGML_TYPE_Q8_0, GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_Q2_0}) {
            f.run(type, 64, true, 8, false, 42); // MMQ, repeated packets and tail
            f.run(type, 512, true, 32, false, 43); // MMB prefill / up
            f.run(type, 512, false, 16, true, 44); // MMB down, empty resident packet
        }
        ++tested; gpu += hip;
    }
    if (!gpu) printf("RDNA3 HIP devices unavailable: GPU kernel tests SKIPPED\n");
    GGML_ASSERT(tested > 0);
    ggml_quantize_free();
}
