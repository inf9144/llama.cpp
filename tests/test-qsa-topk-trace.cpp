#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "hash/hash.h"
#include "json.h"
#include "llama.h"
#include "speculative.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = common_json;

struct row_info {
    std::string score_hash;
    std::string selected_hash;
    bool ambiguous;
    int ties;
};

// Ignore choices among masked pools: they cannot change the QSA attention mask.
static row_info inspect_row(const float * scores, const int32_t * selected, int cols, int k) {
    std::vector<float> ordered(scores, scores + cols);
    for (float v : ordered) {
        if (std::isnan(v) || v == INFINITY) throw std::runtime_error("Invalid indexer score");
    }
    std::nth_element(ordered.begin(), ordered.begin() + k - 1, ordered.end(), std::greater<float>());
    const float cutoff = ordered[k - 1];
    int greater = 0, ties = 0;
    for (float v : ordered) {
        greater += v > cutoff;
        ties += v == cutoff;
    }
    std::vector<int32_t> ids(selected, selected + k), visible;
    std::sort(ids.begin(), ids.end());
    if (ids.front() < 0 || ids.back() >= cols || std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
        throw std::runtime_error("Invalid or duplicate top-k index");
    }
    int selected_greater = 0;
    for (int32_t id : ids) {
        if (scores[id] < cutoff) throw std::runtime_error("Selected score is below the top-k cutoff");
        selected_greater += scores[id] > cutoff;
        if (std::isfinite(scores[id])) visible.push_back(id);
    }
    if (selected_greater != greater) throw std::runtime_error("Top-k omits a strictly greater score");
    return {hash_sha256_hex(scores, size_t(cols)*sizeof(float)),
            hash_sha256_hex(visible.data(), visible.size()*sizeof(int32_t)),
            std::isfinite(cutoff) && ties > k - greater, ties};
}

static std::vector<row_info> inspect(const std::vector<float> & scores, const std::vector<int32_t> & ids, int cols, int k) {
    std::vector<row_info> rows;
    for (size_t r = 0; r < ids.size()/k; ++r) {
        rows.push_back(inspect_row(scores.data() + r*cols, ids.data() + r*k, cols, k));
    }
    return rows;
}

static json frozen_replay(ggml_backend_dev_t dev, const std::vector<float> & scores,
                          const std::vector<row_info> & original, int cols, int k) {
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    if (!backend) throw std::runtime_error("Cannot initialize frozen-replay backend");
    ggml_context_ptr ctx(ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true}));
    if (!ctx) throw std::runtime_error("Cannot allocate frozen-replay context");
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cols, original.size());
    auto * output = ggml_top_k(ctx.get(), input, k);
    ggml_set_input(input);
    ggml_set_output(output);
    if (!ggml_backend_supports_op(backend.get(), output)) throw std::runtime_error("Replay backend does not support TOP_K");
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!buffer) throw std::runtime_error("Cannot allocate frozen-replay tensors");
    ggml_backend_tensor_set(input, scores.data(), 0, ggml_nbytes(input));
    json result = {{"device", ggml_backend_dev_name(dev)}, {"repetitions", json::array()}};
    std::vector<int32_t> ids(ggml_nelements(output));
    std::vector<float> after(scores.size());
    auto previous = original;
    for (int repeat = 0; repeat < 8; ++repeat) {
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Frozen TOP_K execution failed");
        }
        ggml_backend_synchronize(backend.get());
        ggml_backend_tensor_get(output, ids.data(), 0, ggml_nbytes(output));
        ggml_backend_tensor_get(input, after.data(), 0, ggml_nbytes(input));
        if (std::memcmp(scores.data(), after.data(), ggml_nbytes(input)) != 0) {
            throw std::runtime_error("Frozen TOP_K input changed");
        }
        auto rows = inspect(scores, ids, cols, k);
        int changed_original = 0, changed_previous = 0;
        for (size_t r = 0; r < rows.size(); ++r) {
            changed_original += rows[r].selected_hash != original[r].selected_hash;
            changed_previous += rows[r].selected_hash != previous[r].selected_hash;
        }
        result["repetitions"].push_back({{"repeat", repeat}, {"changed_visible_sets_from_original", changed_original},
                                         {"changed_visible_sets_from_previous", changed_previous}});
        previous = std::move(rows);
    }
    result["input_unchanged"] = true;
    return result;
}

struct snapshot {
    std::string name;
    int position, cols, k;
    std::vector<row_info> rows;
};

struct trace_state {
    bool active = false;
    int pass = 0, position = 0;
    size_t call = 0;
    std::string failure;
    std::vector<snapshot> first;
    std::set<ggml_backend_dev_t> replayed;

    static bool callback(ggml_tensor * t, bool ask, void * userdata) {
        auto & state = *static_cast<trace_state *>(userdata);
        if (ask) return state.active && state.failure.empty() && t->op == GGML_OP_TOP_K && std::strncmp(t->name, "indexer_top_k", 13) == 0;
        try {
            state.collect(t);
            return true;
        } catch (const std::exception & e) {
            state.failure = e.what();
            return true;
        }
    }

    json collect(ggml_tensor * t) {
        const auto * src = t->src[0];
        if (!src || !src->buffer || !t->buffer || src->type != GGML_TYPE_F32 || t->type != GGML_TYPE_I32 || !ggml_is_contiguous(src) || !ggml_is_contiguous(t) ||
                src->ne[0] <= 0 || src->ne[0] > 65536 || ggml_nrows(src) <= 0 || ggml_nrows(src) > 1024 || t->ne[0] <= 0 || t->ne[0] > src->ne[0] ||
                ggml_nrows(src) != ggml_nrows(t) || ggml_nbytes(src) > 32*1024*1024) {
            throw std::runtime_error("Unsupported diagnostic TOP_K shape");
        }
        const int cols = int(src->ne[0]), k = int(t->ne[0]);
        std::vector<float> scores(ggml_nelements(src));
        std::vector<int32_t> ids(ggml_nelements(t));
        ggml_backend_tensor_get(src, scores.data(), 0, ggml_nbytes(src));
        ggml_backend_tensor_get(t, ids.data(), 0, ggml_nbytes(t));
        auto rows = inspect(scores, ids, cols, k);
        int ambiguous = 0, max_ties = 0, changed_scores = 0, changed_sets = 0, same_score_changed_set = 0;
        for (const auto & r : rows) {
            ambiguous += r.ambiguous;
            if (r.ambiguous) max_ties = std::max(max_ties, r.ties);
        }
        if (pass == 0) {
            first.push_back({t->name, position, cols, k, rows});
        } else {
            if (call >= first.size()) throw std::runtime_error("Unexpected additional TOP_K call");
            const auto & before = first[call];
            if (before.name != t->name || before.position != position || before.cols != cols || before.k != k || before.rows.size() != rows.size()) {
                throw std::runtime_error("TOP_K trace topology differs between passes");
            }
            for (size_t r = 0; r < rows.size(); ++r) {
                const bool same_score = rows[r].score_hash == before.rows[r].score_hash;
                const bool same_set = rows[r].selected_hash == before.rows[r].selected_hash;
                changed_scores += !same_score;
                changed_sets += !same_set;
                same_score_changed_set += same_score && !same_set;
            }
        }
        json event = {{"pass", pass}, {"position", position}, {"name", t->name}, {"cols", cols}, {"rows", rows.size()}, {"k", k},
                      {"finite_cutoff_ambiguous_rows", ambiguous}, {"max_finite_cutoff_ties", max_ties}};
        if (pass != 0) {
            event["changed_score_rows"] = changed_scores;
            event["changed_visible_set_rows"] = changed_sets;
            event["matching_score_hash_changed_visible_set_rows"] = same_score_changed_set;
        }
        auto * dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(src->buffer));
        if (!dev && ggml_backend_buffer_is_host(src->buffer)) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!dev) throw std::runtime_error("Cannot identify TOP_K buffer device");
        event["device"] = ggml_backend_dev_name(dev);
        if (ambiguous && replayed.insert(dev).second) event["frozen_replay"] = frozen_replay(dev, scores, rows, cols, k);
        std::printf("QSA_TOPK %s\n", event.dump().c_str());
        std::fflush(stdout);
        ++call;
        return event;
    }
};

static void self_test() {
    const float scores[] = {3, 2, 2, 2, 0, -INFINITY, -INFINITY};
    const int32_t a[] = {0, 1}, b[] = {3, 0}, reordered[] = {1, 0};
    const auto left = inspect_row(scores, a, 7, 2);
    const auto right = inspect_row(scores, b, 7, 2);
    GGML_ASSERT(left.ambiguous && left.ties == 3 && left.score_hash == right.score_hash && left.selected_hash != right.selected_hash);
    GGML_ASSERT(left.selected_hash == inspect_row(scores, reordered, 7, 2).selected_hash);
    const int32_t masked_a[] = {0, 1, 2, 3, 4, 5}, masked_b[] = {0, 1, 2, 3, 4, 6};
    GGML_ASSERT(!inspect_row(scores, masked_a, 7, 6).ambiguous);
    GGML_ASSERT(inspect_row(scores, masked_a, 7, 6).selected_hash == inspect_row(scores, masked_b, 7, 6).selected_hash);
    for (auto invalid : {std::vector<int32_t>{1, 2}, {0, 0}, {0, 7}, {0, 4}}) {
        bool rejected = false;
        try { inspect_row(scores, invalid.data(), 7, 2); } catch (const std::runtime_error &) { rejected = true; }
        GGML_ASSERT(rejected);
    }
    const float zero_ties[] = {1, -0.0f, 0.0f};
    const int32_t zero_a[] = {0, 1}, zero_b[] = {0, 2};
    GGML_ASSERT(inspect_row(zero_ties, zero_a, 3, 2).ambiguous);
    GGML_ASSERT(inspect_row(zero_ties, zero_a, 3, 2).selected_hash != inspect_row(zero_ties, zero_b, 3, 2).selected_hash);
    const float all_masked[] = {-INFINITY, -INFINITY, -INFINITY};
    GGML_ASSERT(!inspect_row(all_masked, zero_a, 3, 2).ambiguous);
    GGML_ASSERT(inspect_row(all_masked, zero_a, 3, 2).selected_hash == inspect_row(all_masked, zero_b, 3, 2).selected_hash);
    for (float invalid : {NAN, INFINITY}) {
        const float invalid_scores[] = {3, invalid, 1};
        bool rejected = false;
        try { inspect_row(invalid_scores, zero_a, 3, 2); } catch (const std::runtime_error &) { rejected = true; }
        GGML_ASSERT(rejected);
    }
    ggml_backend_load_all();
    if (auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
        ggml_backend_ptr backend(ggml_backend_dev_init(cpu, nullptr));
        ggml_context_ptr ctx(ggml_init({16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true}));
        auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 7, 2);
        auto * output = ggml_top_k(ctx.get(), input, 2);
        ggml_set_name(output, "indexer_top_k-3");
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
        GGML_ASSERT(buffer);
        std::vector<float> data(scores, scores + 7);
        data.insert(data.end(), {3, 2, 1, 0, -1, -INFINITY, -INFINITY});
        ggml_backend_tensor_set(input, data.data(), 0, ggml_nbytes(input));
        auto * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, output);
        GGML_ASSERT(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS);
        trace_state trace;
        GGML_ASSERT(!trace_state::callback(output, true, &trace));
        trace.active = true;
        GGML_ASSERT(trace_state::callback(output, true, &trace));
        const auto first = trace.collect(output);
        GGML_ASSERT(first["finite_cutoff_ambiguous_rows"].get<int>() == 1);
        const auto & replay = first["frozen_replay"];
        GGML_ASSERT(replay["input_unchanged"].get<bool>());
        for (size_t i = 1; i < replay["repetitions"].size(); ++i) {
            GGML_ASSERT(replay["repetitions"][i]["changed_visible_sets_from_previous"].get<int>() == 0);
        }
        trace.pass = 1;
        trace.call = 0;
        const auto identical = trace.collect(output);
        GGML_ASSERT(identical["changed_score_rows"].get<int>() == 0 && identical["changed_visible_set_rows"].get<int>() == 0);
        std::vector<int32_t> altered(4);
        ggml_backend_tensor_get(output, altered.data(), 0, ggml_nbytes(output));
        const size_t tie_slot = altered[0] == 0 ? 1 : 0;
        altered[tie_slot] = altered[tie_slot] == 1 ? 3 : 1;
        ggml_backend_tensor_set(output, altered.data(), 0, ggml_nbytes(output));
        trace.call = 0;
        const auto changed_selection = trace.collect(output);
        GGML_ASSERT(changed_selection["matching_score_hash_changed_visible_set_rows"].get<int>() == 1);
        data[4] = -0.5f;
        ggml_backend_tensor_set(input, data.data(), 0, ggml_nbytes(input));
        trace.call = 0;
        const auto changed_input = trace.collect(output);
        GGML_ASSERT(changed_input["changed_score_rows"].get<int>() == 1 && changed_input["matching_score_hash_changed_visible_set_rows"].get<int>() == 0);
        std::printf("QSA TopK trace: CPU collection, eight frozen-input replays and input/selection classification validated\n");
    }
    std::printf("QSA TopK trace: finite cutoff ties, masked ties, order independence and invalid selections checked\n");
}

int main(int argc, char ** argv) {
    if (argc == 1) {
        self_test();
        std::printf("Model diagnostic: %s --tokens prompt-tokens.json [existing server model/cache arguments]\n", argv[0]);
        return 0;
    }
    try {
        std::string token_file;
        std::vector<char *> args = {argv[0]};
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--tokens") == 0) {
                if (!token_file.empty() || ++i == argc) throw std::runtime_error("Expected one --tokens file");
                token_file = argv[i];
            } else args.push_back(argv[i]);
        }
        args.push_back(nullptr);
        common_init();
        common_params params;
        if (!common_params_parse(int(args.size()) - 1, args.data(), params, LLAMA_EXAMPLE_SERVER)) return 1;
        if (token_file.empty() || params.n_parallel != 1 || common_speculative_n_max(&params.speculative) != 0) {
            throw std::runtime_error("Requires --tokens, --parallel 1 and draft maximum 0");
        }
        std::ifstream input(token_file, std::ios::binary | std::ios::ate);
        if (!input || input.tellg() <= 0 || input.tellg() > 1024*1024) throw std::runtime_error("Invalid token file size");
        input.seekg(0);
        const json ids = json::parse(std::string(std::istreambuf_iterator<char>(input), {}));
        if (!ids.is_array() || ids.empty() || ids.size() > 8192) throw std::runtime_error("Expected 1..8192 prompt token IDs");
        std::vector<llama_token> tokens;
        for (const auto & id : ids) {
            if (!id.is_number_integer() || id.get<int64_t>() < 0 || id.get<int64_t>() > INT32_MAX) throw std::runtime_error("Invalid token ID");
            tokens.push_back(id.get<llama_token>());
        }
        // Server argument parsing defers model resolution to its caller.
        auto models = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
        common_models_handler_apply(models, params);
        if (params.model.path.empty()) throw std::runtime_error("No resolved target model path");
        if (params.speculative.has_dft() && params.speculative.draft.mparams.path.empty()) {
            throw std::runtime_error("No resolved draft model path");
        }
        trace_state trace;
        params.cb_eval = trace_state::callback;
        params.cb_eval_user_data = &trace;
        params.warmup = false;
        params.sampling.backend_sampling = false;
        llama_backend_init();
        llama_numa_init(params.numa);
        auto init = common_init_from_params(params);
        auto * ctx = init->context();
        auto * model = init->model();
        if (!ctx || !model) throw std::runtime_error("Cannot initialize target model/context");
        char architecture[32] = {};
        llama_model_meta_val_str(model, "general.architecture", architecture, sizeof(architecture));
        if (std::strcmp(architecture, "qwen4exp") != 0) throw std::runtime_error("This diagnostic requires a qwen4exp model");
        const auto * vocab = llama_model_get_vocab(model);
        for (auto token : tokens) if (token >= llama_vocab_n_tokens(vocab)) throw std::runtime_error("Token ID exceeds vocabulary");
        if (tokens.size() > llama_n_ctx(ctx)) throw std::runtime_error("Prompt exceeds context");

        common_speculative_init_result_ptr draft;
        common_speculative_ptr spec;
        if (params.speculative.has_dft()) {
            auto draft_params = common_base_params_to_speculative(params);
            draft_params.cb_eval = nullptr;
            draft_params.cb_eval_user_data = nullptr;
            draft = common_speculative_init_from_params(draft_params, model, ctx);
            if (!draft->model() || !draft->context()) throw std::runtime_error("Cannot initialize configured draft");
            params.speculative.draft.ctx_tgt = ctx;
            params.speculative.draft.ctx_dft = draft->context();
            spec.reset(common_speculative_init(params.speculative, 1));
            if (!spec) throw std::runtime_error("Cannot initialize speculative feature capture");
        }
        const int width = std::min<int>({int(llama_n_batch(ctx)), int(llama_n_ubatch(ctx)), 1024});
        trace.active = true;
        for (int pass = 0; pass < 2; ++pass) {
            llama_synchronize(ctx);
            llama_memory_clear(llama_get_memory(ctx), true);
            trace.pass = pass;
            trace.call = 0;
            for (size_t pos = 0; pos < tokens.size(); pos += width) {
                trace.position = int(pos);
                common_batch batch(ctx);
                const size_t end = std::min(pos + width, tokens.size());
                for (size_t i = pos; i < end; ++i) batch.add(tokens[i], int(i), 0, i + 1 == end);
                if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0 || !trace.failure.empty()) {
                    throw std::runtime_error("Prefill trace failed: " + trace.failure);
                }
            }
            llama_synchronize(ctx);
            if (trace.call == 0 || trace.call != trace.first.size()) throw std::runtime_error("Missing or inconsistent QSA TopK coverage");
        }
        trace.active = false;
        std::printf("QSA_TOPK_DONE %s\n", json({{"complete", true}, {"prompt_tokens", tokens.size()}, {"calls_per_pass", trace.first.size()},
                                             {"draft_feature_capture", bool(spec)}, {"note", "Instrumented prefill and frozen TOP_K only; no decode, verify or timing conclusions"}}).dump().c_str());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "QSA TopK trace failed: %s\n", e.what());
        return 1;
    }
    return 0;
}
