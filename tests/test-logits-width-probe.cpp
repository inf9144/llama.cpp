// Compare the same query rows in decode and speculative verify batches (widths 1..8).
// Adapted from stew675/llama.cpp, block 0006. No model or backend settings come from the environment.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

static void quiet_log(ggml_log_level, const char *, void *) {}

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 6) {
        fprintf(stderr, "usage: %s model.gguf text.txt [prefill=4096] [ubatch=1024] [kv=q8_0]\n", argv[0]);
        return 1;
    }
    constexpr int widths = 8;
    const int prefix = argc > 3 ? atoi(argv[3]) : 4096;
    const int ubatch = argc > 4 ? atoi(argv[4]) : 1024;
    const std::string kv_name = argc > 5 ? argv[5] : "q8_0";
    const ggml_type kv = kv_name == "f16" ? GGML_TYPE_F16 : kv_name == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_COUNT;
    if (prefix < 1 || prefix > 262144 || ubatch < widths || ubatch > 262144 || kv == GGML_TYPE_COUNT) {
        fprintf(stderr, "invalid dimensions or KV type (supported: f16, q8_0)\n");
        return 1;
    }
    std::ifstream input(argv[2], std::ios::binary);
    if (!input) {
        fprintf(stderr, "cannot open text\n");
        return 1;
    }
    const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (text.size() > 64 * 1024 * 1024) {
        fprintf(stderr, "text exceeds 64 MiB\n");
        return 1;
    }
    llama_log_set(quiet_log, nullptr);
    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "model load failed\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(text.size() + 1);
    const int count = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), false, false);
    if (count < prefix + widths) {
        fprintf(stderr, "need at least %d tokens (got %d)\n", prefix + widths, count);
        llama_model_free(model);
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<std::vector<float>> rows[widths];
    bool ok = true;
    for (int width = 1; width <= widths && ok; ++width) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = prefix + widths + 8;
        cp.n_batch = std::max(prefix, ubatch);
        cp.n_ubatch = ubatch;
        cp.n_seq_max = 1;
        cp.n_rs_seq = width - 1;
        cp.n_rs_batch = widths;
        cp.type_k = kv;
        cp.type_v = kv;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) {
            ok = false;
            break;
        }
        llama_batch batch = llama_batch_init(cp.n_batch, 0, 1);
        auto fill = [&](int begin, int n, bool logits) {
            batch.n_tokens = n;
            for (int i = 0; i < n; ++i) {
                batch.token[i] = tokens[begin + i];
                batch.pos[i] = begin + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = logits;
            }
        };
        fill(0, prefix, false);
        ok = llama_decode(ctx, batch) == 0;
        if (ok) {
            fill(prefix, width, true);
            ok = llama_decode(ctx, batch) == 0;
        }
        for (int j = 0; j < width && ok; ++j) {
            const float * logits = llama_get_logits_ith(ctx, j);
            ok = logits != nullptr;
            if (ok) {
                rows[width - 1].emplace_back(logits, logits + n_vocab);
            }
        }
        llama_batch_free(batch);
        llama_free(ctx);
    }
    bool identical = ok;
    if (ok) {
        for (int width = 1; width <= widths; ++width) {
            float max_diff = 0;
            bool same = true;
            for (int row = 0; row < width; ++row) {
                const auto & a = rows[width - 1][row];
                const auto & b = rows[widths - 1][row];
                same &= memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
                for (int i = 0; i < n_vocab; ++i) {
                    identical &= std::isfinite(a[i]) && std::isfinite(b[i]);
                    max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
                }
            }
            printf("width=%d, maxdiff=%g, bit_identical=%s\n", width, max_diff, same ? "yes" : "no");
            identical &= same;
        }
    }
    llama_model_free(model);
    llama_backend_free();
    printf("width_purity=%s\n", identical ? "PASS" : "FAIL");
    return !ok ? 1 : identical ? 0 : 2;
}
