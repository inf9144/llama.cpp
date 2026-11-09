#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "llama.h"
#include "llama-cpp.h"
#ifdef LLAMA_TEST_FORK_INTERNALS
#include "speculative.h"

#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-context.h"
#include "../src/llama-memory.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-model.h"
#include "../src/llama-moe-cache.h"
#include "../src/llama-moe-cache-policy.h"
#include "../ggml/src/ggml-moe.h"
#endif

#include "../src/llama-ext.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <limits>
#include <map>
#include <filesystem>
#include <memory>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <locale>
#endif

static bool arch_matches(const std::string & filter, llm_arch arch) {
    if (filter.empty()) {
        return true;
    }
    return std::regex_search(llama_arch_name(arch), std::regex(filter));
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

struct tensor_data_params {
    size_t seed;
    float  stdev;
};

// fills the values of one tensor, the name carries whatever the property needs
typedef void (*tensor_filler)(std::vector<float> & dst, const char * name, const tensor_data_params & params);

// a pool of normal draws, generated once and reused for every model
static std::vector<float> g_normal_pool;
static const size_t NORMAL_POOL_SIZE = 16 * 1024 * 1024;

// fill the pool once; each tensor copies a slice of it
static void init_normal_pool(const size_t seed, const float stdev) {
    g_normal_pool.resize(NORMAL_POOL_SIZE);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, stdev);
    for (float & v : g_normal_pool) {
        v = dist(gen);
    }
}

// a normal draw keyed on a name, so that tensors which share a draw ask for the same key
static void fill_normal(std::vector<float> & dst, const char * key, const tensor_data_params &) {
    const size_t n = dst.size();
    const size_t pool_size = g_normal_pool.size();
    const size_t start = std::hash<std::string>{}(key) % pool_size;

    // copy a cyclic slice of the pool, so a tensor larger than the pool still fills
    size_t i = 0;
    size_t pos = start;
    while (i < n) {
        const size_t chunk = std::min(pool_size - pos, n - i);
        std::memcpy(dst.data() + i, g_normal_pool.data() + pos, chunk * sizeof(float));
        i += chunk;
        pos = 0;
    }
}

// Mamba A must be negative (state decay)
static void fill_ssm_a(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    fill_normal(dst, name, params);
    for (float & v : dst) {
        v = -fabsf(v);
    }
}

// RoPE factors divide the frequencies, a value close to 0 gives angles too large for F32 sin/cos to stay accurate
static void fill_rope_factors(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    fill_normal(dst, name, params);
    for (float & v : dst) {
        v = 1.0f + fabsf(v);
    }
}

// FFN act scales divide the activation, a value close to 0 overflows the F16 staging of the Metal mul-mat kernel
static void fill_ffn_act(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    fill_normal(dst, name, params);
    for (float & v : dst) {
        v = copysignf(1.0f + fabsf(v), v);
    }
}

// a norm preserving residual gate holds alpha^2 + beta^2 == 1, and the conversion resolves both from one raw weight
// the alpha and the beta of a block share that weight, so the key drops the suffix and keeps the block
static void fill_resid_gate(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    const char * suffix = strstr(name, "_resid_");
    const bool is_beta = strstr(suffix, "_resid_beta") != nullptr;
    fill_normal(dst, std::string(name, suffix - name).c_str(), params);
    for (float & v : dst) {
        const float beta = 1.0f / (1.0f + expf(-v));
        v = is_beta ? beta : sqrtf(1.0f - beta*beta);
    }
}

// a random router bias can tie within the noise between the backends, a ramp keeps the top-k selection the same
static void fill_exp_probs_b(std::vector<float> & dst, const char *, const tensor_data_params &) {
    for (size_t i = 0; i < dst.size(); i++) {
        dst[i] = 2.0f*i;
    }
}

// a tensor whose values hold a property of their own, matched on a substring of its name, first match wins
static const struct { const char * match; tensor_filler fill; } TENSOR_FILLERS[] = {
    { "ssm_a",        fill_ssm_a        },
    { "_resid_",      fill_resid_gate   },
    { "exp_probs_b",  fill_exp_probs_b  },
    { "ffn.act",      fill_ffn_act      },
    { "rope_freqs",   fill_rope_factors },
    { "rope_factors", fill_rope_factors },
};

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    const tensor_data_params & params = *(const tensor_data_params *) userdata;

    tensor_filler fill = fill_normal;
    for (const auto & filler : TENSOR_FILLERS) {
        if (strstr(tensor->name, filler.match) != nullptr) {
            fill = filler.fill;
            break;
        }
    }

    std::vector<float> tmp(ggml_nelements(tensor));
    fill(tmp, tensor->name, params);

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp16(tmp.size());
        for (size_t i = 0; i < tmp.size(); i++) {
            tmp16[i] = ggml_fp32_to_fp16(tmp[i]);
        }
        ggml_backend_tensor_set(tensor, tmp16.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void usage(char ** argv) {
    LOG("Usage: %s [options]\n\n", argv[0]);
    LOG("Options:\n");
    LOG("  -a, --arch <arch|regex>  Run only matching LLM architectures (default: all supported)\n");
    LOG("  -s, --seed <seed>        Set the random seed for tensor initialization and token generation\n");
    LOG("  -d, --stdev <stdev>      Set the standard deviation of the tensor initialization distribution (default: 0.1f)\n");
    LOG("  -o, --out <dir>          Save generated test models to <dir> instead of running backend tests\n");
    LOG("  -v <N>                   Set log verbosity level\n");
    LOG("  -b, --backend <backend>  Run only on the given backend device\n");
    LOG("  --test-graph-recovery    Test allocation recovery with small CPU models\n");
    LOG("  --test-moe-cache-prefill Test RDNA3 cache locality, prefill logits and checkpoints\n");
    LOG("  --test-moe-cache-policy  Test shared SLRU prefill admission and capacity bounds\n");
    LOG("  --test-layer-inputs      Test Qwen4Exp layer extraction and MoE cache parameters\n");
    LOG("  --test-dflash-device-inputs  Compare device-gather and host DFlash2 injection\n");
    LOG("  --test-dflash-selector   Check compact DFlash2 scores across two devices\n");
    LOG("  --test-dflash-vision     Test DFlash2 drafting across M-RoPE image positions\n");
    LOG("  --test-kq-derived       Compare compact mask inputs with the packed mask fill\n");
    LOG("  --test-dflash-tail       Test DFlash2 prompt tails, cache reuse and SWA guards\n");
    LOG("  --kv-shift               Check K-shift against a decode at the shifted positions (CPU only)\n");
    LOG("  -h, --help               Show this help message\n\n");
    LOG("Examples:\n");
    LOG("  %s\n", argv[0]);
    LOG("  %s -a qwen35moe\n", argv[0]);
    LOG("  %s -a deepseek4 -o tests/test-models/\n", argv[0]);
    LOG("  %s -a cohere2moe -v 5\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

// MoE archs that are also tested with the experts in host memory
static bool host_experts_test(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_DEEPSEEK2:
            return true;
        default:
            return false;
    }
}

// writes arch-resolved GGUF keys into the fixture context, the test-side equivalent of llama_model_saver
struct kv_writer {
    struct gguf_context * gguf_ctx;
    const llm_arch        arch;

    kv_writer(enum llm_arch arch, struct gguf_context * gguf_ctx) : gguf_ctx(gguf_ctx), arch(arch) {}

    void add_kv(enum llm_kv key, uint32_t     value) { gguf_set_val_u32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, int32_t      value) { gguf_set_val_i32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, uint64_t     value) { gguf_set_val_u64 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, float        value) { gguf_set_val_f32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, bool         value) { gguf_set_val_bool(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, const char * value) { gguf_set_val_str (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }

    template <typename T>
    void add_kv(enum llm_kv key, const std::vector<T> & value) {
        if (value.empty()) {
            return;
        }
        if (std::is_same<T, uint32_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_UINT32, value.data(), value.size());
        } else if (std::is_same<T, int32_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_INT32, value.data(), value.size());
        } else if (std::is_same<T, uint64_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_UINT64, value.data(), value.size());
        } else if (std::is_same<T, float>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_FLOAT32, value.data(), value.size());
        } else {
            GGML_ABORT("fatal error");
        }
    }

    void add_kv(enum llm_kv key, const std::vector<std::string> & value) {
        std::vector<const char *> tmp(value.size());
        for (size_t i = 0; i < value.size(); ++i) {
            tmp[i] = value[i].c_str();
        }
        gguf_set_arr_str(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), tmp.data(), tmp.size());
    }
};

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe, const uint32_t n_layer_min = 0, uint32_t n_vocab = 128) {
    gguf_context_ptr ret(gguf_init_empty());
    kv_writer ms(arch, ret.get());
    const uint32_t n_ctx = 256;

    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_MAION_CODER) {
        n_head = 4;
        n_layer = 4; // dense and MoE layers, each with local and full attention
    } else if (arch == LLM_ARCH_GEMMA4) {
        // TODO: expand coverage to both GQA and MQA (see https://github.com/ggml-org/llama.cpp/pull/31401#pullrequestreview-5710165130)
        n_head = 4;
        n_layer = 10; // last 5 layers reuse KV, both SWA and full attention
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_LFM3) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_DEEPSEEK4) {
        // head size 64 so that GPU flash attention kernels support the model
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 4;
    } else if (arch == LLM_ARCH_DEEPSEEK41) {
        // window-only, Engram, ratio-2 and ratio-1 sources, top-k reuse and a candidate-masked index layer
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 6;
    } else if (arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_LAGUNA) {
        n_embd = 160; // exercise per-head tensor split granularity with head size 80
    } else if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE || arch == LLM_ARCH_RICURSIVE) {
        n_head = 4;
    } else if (arch == LLM_ARCH_LIMITE) {
        n_head  = 4;
        n_layer = 4; // dense connections on the last two layers
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_KIMI_K31
            || arch == LLM_ARCH_MISTRAL4
            || arch == LLM_ARCH_MISTRAL_LARGE4
            || arch == LLM_ARCH_HY_V4
            || arch == LLM_ARCH_XING4_0) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
        if (arch == LLM_ARCH_MISTRAL_LARGE4) {
            n_layer = 4; // cover NoPE, shared indexer and KV sharing layers
        }
    } else if (arch == LLM_ARCH_GLM5_NEXT) {
        // several heads so that -sm tensor can split them and the MLA flash attention has GQA > 1
        n_embd = 128;
        n_head = 16;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_METIS16) {
        n_layer = 4; // 2 passes over 2 physical layers, one attention and one Mamba-2
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    } else if (arch == LLM_ARCH_QWEN3TTS) {
        //n_vocab = 4096; // must be >= the hard-coded codec head size (3072)
        n_vocab = 3072; // TODO: should be 4096, but user code cannot get `n_vocab_out` yet [TAG_LLAMA_N_VOCAB_OUT]
    } else if (arch == LLM_ARCH_HRM_TEXT) {
        n_layer = 6; // 1 layer per stack x 2 h-cycles x (2 l-cycles + 1) cache slots
    }

    uint32_t n_head_kv = n_head;
    if (arch == LLM_ARCH_QWEN3) {
        n_head_kv = 1; // MQA coverage
    } else if (arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_MUSE_SPARK || arch == LLM_ARCH_AFMOE || arch == LLM_ARCH_MAION_CODER ||
            arch == LLM_ARCH_KOLIBRI1 || arch == LLM_ARCH_LIMITE || arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_RICURSIVE) {
        n_head_kv = 2; // GQA coverage
    }
    if (arch == LLM_ARCH_BEAM) {
        n_head    = 4;
        n_head_kv = 2;
        n_layer   = 4;
    }
    n_layer = std::max(n_layer, n_layer_min);
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llama_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));

    if (arch == LLM_ARCH_K2_HORIZON) {
        ms.add_kv(LLM_KV_ROPE_SCALING_YARN_BETA_FAST, 128.0f);
        ms.add_kv(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,   4.0f);
    }

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_LFM3 ||
            arch == LLM_ARCH_KIMI_LINEAR || arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_KIMI_K3 || arch == LLM_ARCH_KIMI_K31 ||
            arch == LLM_ARCH_GLM5_NEXT
            ) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        // GLM5 next KDA heads come from the uniform head count, only head_count_kv is per layer.
        if (arch == LLM_ARCH_GLM5_NEXT) {
            ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
            // the MLA cache holds one latent head
            for (uint32_t & n : n_head_per_layer) {
                n = std::min(n, uint32_t(1));
            }
        } else {
            ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else if (arch == LLM_ARCH_METIS16) {
        // the layer types repeat in every pass
        std::vector<uint32_t> n_head_kv_per_layer;
        n_head_kv_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_kv_per_layer.push_back(il % 2 == 0 ? n_head_kv : 0);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_kv_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) ? uint32_t(1) : n_head_kv);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,   n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, n_embd_head);
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,   n_embd_head/2);
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_KIMI_K31
            || arch == LLM_ARCH_GLM5_NEXT
            || arch == LLM_ARCH_HY_V4
            || arch == LLM_ARCH_MISTRAL4
            || arch == LLM_ARCH_MISTRAL_LARGE4
            || arch == LLM_ARCH_XING4_0) {
        // GLM5 next MLA is nope only, the cache row is the compressed latent alone.
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       arch == LLM_ARCH_GLM5_NEXT ? uint32_t(512) : uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       arch == LLM_ARCH_GLM5_NEXT ? uint32_t(0) : uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
        if (arch == LLM_ARCH_DOTS3NOTE) {
            // SWA layers reuse the same MLA geometry as the full layers in this fixture
            ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,       uint32_t(576));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA_SWA,   uint32_t(192));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA_SWA, uint32_t(128));
            ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,             10000.0f);
            // indexer on the full-attention layers (inverse of the swa pattern)
            std::vector<uint32_t> indexer_types;
            indexer_types.reserve(n_layer);
            for (uint32_t il = 0; il < n_layer; il++) {
                indexer_types.push_back(il % 2 ? 0 : 1);
            }
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
        }
    } else if (arch == LLM_ARCH_MINIMAX_M3) {
        // partial rotary: n_rot must not exceed the indexer key length (64)
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
    }
    if (arch == LLM_ARCH_SENSENOVA_U1) {
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SPATIAL, 10000.0f);
    }

    if (arch == LLM_ARCH_XING4_0) {
        // Manifold-Constrained Hyper-Connections (MHC) multi-residual-stream blocks
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(20));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1e-6f);
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) ? uint32_t(64) : uint32_t(512));
    if (arch != LLM_ARCH_DFLASH) {
        ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK, uint32_t(512));
    }
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_MAION_CODER) {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, std::vector<uint32_t>{1, 1, 0, 0});
    } else if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(5));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back((il + 1) % 5 != 0);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_SPARK2_5 ||
            arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_MUSE_SPARK || arch == LLM_ARCH_GRANITE_SWA || arch == LLM_ARCH_DOTS3NOTE ||
            arch == LLM_ARCH_KOLIBRI1 || arch == LLM_ARCH_MAPLE || arch == LLM_ARCH_FLINT || arch == LLM_ARCH_BEAM || arch == LLM_ARCH_RICURSIVE) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
        if (arch == LLM_ARCH_FLINT) {
            ms.add_kv(LLM_KV_ATTENTION_ROPE_PATTERN, pattern);
        }
    } else if (arch == LLM_ARCH_LFM3) {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, std::vector<uint32_t>({0, 0, 1}));
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,               10000.0f);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    if (arch == LLM_ARCH_MUSE_SPARK) {
        ms.add_kv(LLM_KV_POST_NORM_EPS,           1e-5f);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,      10000.0f);
        ms.add_kv(LLM_KV_FINAL_LOGIT_SOFTCAPPING, 30.0f);
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE,         n_embd/2);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,    1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,     true);
    }

    if (arch == LLM_ARCH_KOLIBRI1 || arch == LLM_ARCH_LIMITE) {
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM, false);
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    if (arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_GLM5_NEXT) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,  1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
        // without this the QSA layers fall back to dense and go uncovered
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(n_layer, 4));

        // has_cell_ext() needs ple_n_heads here: the indexer cache serializes no ext without it
        const uint32_t ple_ngram_size      = 3;
        const uint32_t ple_heads_per_ngram = 2;
        const uint32_t ple_n_heads         = (ple_ngram_size - 1)*ple_heads_per_ngram;
        GGML_ASSERT(n_embd % ple_n_heads == 0);
        const uint32_t ple_head_dim = n_embd/ple_n_heads;

        std::vector<uint64_t> ple_head_offsets(ple_n_heads);
        std::vector<uint64_t> ple_head_vocab_sizes(ple_n_heads, n_vocab);
        for (uint32_t h = 0; h < ple_n_heads; h++) {
            ple_head_offsets[h] = uint64_t(h)*n_vocab;
        }

        // the PLE history lives in the recurrent cache, so it must sit on a linear attention layer
        ms.add_kv(LLM_KV_PLE_LAYERS,                  std::vector<uint32_t>({ 0 }));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,              ple_ngram_size);
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,         ple_heads_per_ngram);
        ms.add_kv(LLM_KV_PLE_CONV_KERNEL,             uint32_t(4));
        ms.add_kv(LLM_KV_PLE_EOS_TOKEN_ID,            uint32_t(0));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  ple_head_dim);
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,       std::vector<uint64_t>({ 1, 3, 5 }));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,            ple_head_offsets);
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,        ple_head_vocab_sizes);
    }

    // minimax-m3 keeps one indexer head per GQA head; the rest use a fixed 64 to match the fused
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   arch == LLM_ARCH_MINIMAX_M3 ? n_head : uint32_t(64));
    // qwen4exp ropes indexer keys with the main rotary width, so its head can't be < n_rot
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,
              arch == LLM_ARCH_QWEN4EXP ? n_embd_head : uint32_t(128));

    // note: using a realistic top-k here makes the results unstable and hard to match between CPU and GPU
    //       a large value makes things deterministic since all data is selected by the indexer
    //ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(131072));

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL,        uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL_SELECT_TAIL, true);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    // mrope sections count rope pairs; Ling 3.0 VL files carry [t, h, w] sections
    // summing to n_rot / 2 (n_rot is 64 in this fixture)
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({8, 12, 12, 0}));
    } else {
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    }

    if (arch == LLM_ARCH_HY_V4) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,     uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,   1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_MAGNITUDE, 2.0f);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,           10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,       1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,        true);
        // layer 0 must own an indexer, the odd layers share it
        std::vector<uint32_t> indexer_types;
        indexer_types.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            indexer_types.push_back(il % 2 ? 0 : 1);
        }
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
    }

    if (arch == LLM_ARCH_MISTRAL_LARGE4) {
        // layer 3 reads the KV cache of layer 2, both are NoPE
        ms.add_kv(LLM_KV_ATTENTION_ROPE_PATTERN,     std::vector<uint32_t>({1, 1, 0, 0}));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES,    std::vector<uint32_t>({1, 0, 1, 1}));
        ms.add_kv(LLM_KV_ATTENTION_KV_SOURCE_MAP,    std::vector<uint32_t>({0, 1, 2, 2}));
        ms.add_kv(LLM_KV_FINAL_LOGIT_SOFTCAPPING,    30.0f);
    }

    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,          uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,            uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,             std::vector<uint32_t>({0, 0, 4, 128}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,     160000.0f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,                uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS,  uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,              1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
    }

    if (arch == LLM_ARCH_DEEPSEEK41) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,          uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,            uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,             std::vector<uint32_t>({0, 0, 2, 2, 1, 1}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,     160000.0f);
        ms.add_kv(LLM_KV_ATTENTION_KV_SOURCE_LAYERS,            std::vector<uint32_t>({2, 4}));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_SOURCE_LAYERS,       std::vector<uint32_t>({2, 4, 5}));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_SOURCE_LAYER, int32_t(4));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_TOP_K_BLOCKS, uint32_t(4096));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE,   uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,                uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS,  uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,              1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
        std::vector<int32_t> token_map(n_vocab);
        for (uint32_t i = 0; i < n_vocab; ++i) {
            token_map[i] = (int32_t) (i % 61);
        }
        ms.add_kv(LLM_KV_PLE_LAYERS,                   std::vector<uint32_t>({1, 3}));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,               uint32_t(3));
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,          uint32_t(2));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,   uint32_t(64));
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,        std::vector<uint64_t>({ 76632096046245ull, 4839876093313ull, 35959672319349ull,
                                                                               67716810739261ull, 51510806800915ull, 30921347202721ull }));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,             std::vector<uint64_t>({0, 101, 204, 311, 420, 533, 660, 791}));
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,         std::vector<uint64_t>({101, 103, 107, 109, 113, 127, 131, 137}));
        ms.add_kv(LLM_KV_ENGRAM_PAD_TOKEN_ID,          uint32_t(2));
        ms.add_kv(LLM_KV_ENGRAM_TOKEN_MAP,             token_map);
    }

    if (arch == LLM_ARCH_HRM_TEXT) {
        // 6 cache slots alias 2 physical blocks: 1 low-stack layer + 1 high-stack layer
        ms.add_kv(LLM_KV_HRM_LAYERS_PER_STACK, uint32_t(1));
        ms.add_kv(LLM_KV_HRM_H_CYCLES,         uint32_t(2));
        ms.add_kv(LLM_KV_HRM_L_CYCLES,         uint32_t(2));
    }

    if (arch == LLM_ARCH_MAPLE) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP, 7.0f);
    }

    if (arch == LLM_ARCH_RICURSIVE) {
        std::vector<float> swiglu_clamp;
        swiglu_clamp.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            // the random weight scales keep the FFN activations small, a small limit still clamps so the roundtrip checks the per-layer values
            swiglu_clamp.push_back(il % 3 == 1 ? 0.0f : 1e-3f);
        }
        ms.add_kv(LLM_KV_EMBEDDING_SCALE, 16.0f);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP,    swiglu_clamp);
    }

    if (arch == LLM_ARCH_METIS16) {
        ms.add_kv(LLM_KV_METIS_PASS_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,         uint32_t(4));
        ms.add_kv(LLM_KV_METIS_MEMORY_LENGTH,            uint32_t(64));
        ms.add_kv(LLM_KV_METIS_ROUTE_LENGTH,             uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_PASS_LORA_RANK,       uint32_t(8));
        ms.add_kv(LLM_KV_METIS_CONTINUATION_LOGIT_LIMIT, 4.0f);
        ms.add_kv(LLM_KV_SSM_NORM_EPS,                   1e-5f);
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE,                n_embd/2);
    }

    if (arch == LLM_ARCH_LIMITE) {
        ms.add_kv(LLM_KV_ATTENTION_SCALE,             0.1f);
        ms.add_kv(LLM_KV_ATTENTION_XSA_EPS,           1e-4f);
        ms.add_kv(LLM_KV_FINAL_LOGIT_SIGMOID_CAPPING, std::vector<float>({23.0f, 5.0f, 7.5f}));
        ms.add_kv(LLM_KV_MUDD_FEED_FORWARD_LENGTH,    uint32_t(8));
        ms.add_kv(LLM_KV_MUDD_TAP_COUNT,              uint32_t(2));
        ms.add_kv(LLM_KV_MUDD_TAP_INDICES,            std::vector<int32_t>({-1, -1, -1, -1, 0, 2, 1, 3}));
    }

    // dummy tokenizer: token ids are derived from fixed-size chunks and detokenized as hex ids
    {
        std::vector<std::string> tokenizer_list(n_vocab);
        std::vector<float>       tokenizer_scores(n_vocab, 0.0f);

        ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "test");
        for (uint32_t i = 0; i < n_vocab; i++) {
            tokenizer_list[i] = "tok_" + std::to_string(i);
        }
        ms.add_kv(LLM_KV_TOKENIZER_LIST,   tokenizer_list);
        ms.add_kv(LLM_KV_TOKENIZER_SCORES, tokenizer_scores);
    }

    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);  // distinct from n_ff so a saver key-clobber surfaces on reload
        ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,       n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        // with more experts than a ubatch uses, the copy of the used experts in host memory skips some of them
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(host_experts_test(arch) ? 64 : arch == LLM_ARCH_CLOVER_MOE ? 4 : 2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));

        uint32_t gating_func = 2;
        switch (arch) {
            case LLM_ARCH_MELLUM:
                gating_func = 1;
                break;
            case LLM_ARCH_DEEPSEEK4:
            case LLM_ARCH_DEEPSEEK41:
                gating_func = 4;
                break;
            default:
                break;
        }

        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         gating_func); // sigmoid : softmax : sqrtsoftplus
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
        if (arch == LLM_ARCH_FLINT) {
            ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM, true);
        }
        if (arch == LLM_ARCH_K2_HORIZON) {
            ms.add_kv(LLM_KV_ATTENTION_VALUE_EXPERT_COUNT,      uint32_t(2));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_EXPERT_USED_COUNT, uint32_t(2));
        }
    }

    if (arch == LLM_ARCH_BEAM) {
        ms.add_kv(LLM_KV_EXPERT_COUNT,            uint32_t(8));
        ms.add_kv(LLM_KV_EXPERT_GROUP_COUNT,      uint32_t(4));
        ms.add_kv(LLM_KV_EXPERT_GROUP_USED_COUNT, uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,    2.5f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,     true);
    }

    if (arch == LLM_ARCH_MAION_CODER) {
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE, n_embd / 2);
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, std::vector<uint32_t>{n_ff, n_ff / 2, n_ff, n_ff});
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_CLOVER_MOE ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,             true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);
    if (arch == LLM_ARCH_BAILINGMOE3) {
        std::vector<float> clamp_exp;
        std::vector<float> clamp_shexp;
        for (uint32_t il = 0; il < n_layer; il++) {
            clamp_exp  .push_back(il % 2 ? 4.0f : 0.0f);
            clamp_shexp.push_back(il % 2 ? 5.0f : 0.0f);
        }
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,   clamp_exp);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, clamp_shexp);
    }
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,               n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,           uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,              3.5565588200778455f);
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,         uint32_t(12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,        4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,        -5.0f);

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static std::string get_tmp_gguf_path() {
    static size_t id = 0;
    return (std::filesystem::temp_directory_path() / ("test-llama-archs-" + std::to_string(++id) + ".gguf")).string();
}

static llama_model_ptr get_model(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const float stdev,
        const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER,
        const llama_model_tensor_buft_override * tensor_buft_overrides = nullptr) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;
    model_params.tensor_buft_overrides = tensor_buft_overrides;

    tensor_data_params tensor_params = { seed, stdev };
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tensor_params, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    return model;
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const float stdev,
        const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false,
        const llama_model_tensor_buft_override * tensor_buft_overrides = nullptr, const size_t moe_cache_size = 0,
        const std::vector<size_t> & moe_cache_sizes = {}) {
    llama_model_ptr model = get_model(gguf_ctx, file, seed, stdev, devs, split_mode, tensor_buft_overrides);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    if (!encode) {
        ctx_params.n_ubatch = 64;
    }
    if (moe_cache_size > 0 || !moe_cache_sizes.empty()) {
        // the MoE cache is only used for small ubatches
        ctx_params.moe_cache_size = moe_cache_size;
        ctx_params.moe_cache_split = moe_cache_sizes.empty() ? nullptr : moe_cache_sizes.data();
        ctx_params.n_ubatch = 2;
    }

    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    if (!moe_cache_sizes.empty()) {
        auto uncached_params = ctx_params;
        uncached_params.moe_cache_size = 0;
        uncached_params.moe_cache_split = nullptr;
        llama_context_ptr uncached(llama_init_from_model(model.get(), uncached_params));
        GGML_ASSERT(uncached);
        auto actual = llama_get_memory_breakdown(lctx.get());
        auto baseline = llama_get_memory_breakdown(uncached.get());
        for (size_t i = 0; i < devs.size(); ++i) {
            auto buft = ggml_backend_dev_buffer_type(devs[i]);
            GGML_ASSERT(actual[buft].context >= baseline[buft].context);
            const size_t allocated = actual[buft].context - baseline[buft].context;
            GGML_ASSERT(allocated <= moe_cache_sizes[i]);
            if (moe_cache_sizes[i] >= 1536*1024 && baseline[buft].context > 0) {
                GGML_ASSERT(allocated > 0);
            }
        }
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    common_batch batch(lctx);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        batch.add(tokens[pos], pos, 0, true);
    }
    if (encode) {
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_ENCODE, batch.get())) {
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    return ret;
}

// entries [n/4, n/2) are embd rows, decoded either as token/embd/token chunks or as one mixed batch
// returns the llama_process() error code
static int32_t get_logits_mixed(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, const std::vector<float> & embd, bool mixed,
        std::vector<float> & ret) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_embd   = llama_model_n_embd_inp(model);
    const uint32_t n_tokens = tokens.size();
    const uint32_t i_embd_0 = n_tokens/4;
    const uint32_t i_embd_1 = n_tokens/2;

    const std::vector<uint32_t> bounds = mixed
        ? std::vector<uint32_t>{0, n_tokens}
        : std::vector<uint32_t>{0, i_embd_0, i_embd_1, n_tokens};

    llama_memory_clear(llama_get_memory(lctx), true);
    llama_batch_ext_ptr batch(llama_batch_ext_init(lctx));

    ret.clear();
    ret.reserve(n_tokens*n_vocab);
    for (size_t c = 0; c + 1 < bounds.size(); c++) {
        llama_batch_ext_clear(batch.get());
        for (uint32_t i = bounds[c]; i < bounds[c + 1]; i++) {
            const bool is_embd = i >= i_embd_0 && i < i_embd_1;
            const int32_t idx = is_embd
                ? llama_batch_ext_add_embd(batch.get(), 0, { embd.data() + (size_t) (i - i_embd_0)*n_embd, 1, n_embd })
                : llama_batch_ext_add_token(batch.get(), 0, tokens[i]);
            GGML_ASSERT(idx >= 0);
            const llama_pos pos[4] = { (llama_pos) i, (llama_pos) i, (llama_pos) i, 0 };
            llama_batch_ext_set_pos(batch.get(), idx, pos);
            llama_batch_ext_set_output_logits(batch.get(), idx, true);
        }
        const int32_t err = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (err != 0) {
            return err;
        }
        for (uint32_t i = 0; i < bounds[c + 1] - bounds[c]; i++) {
            const float * logits_ith = llama_get_logits_ith(lctx, i);
            ret.insert(ret.end(), logits_ith, logits_ith + n_vocab);
        }
    }
    return 0;
}

static bool check_causal_attn_toggle(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_past   = tokens.size();
    const uint32_t n_ubatch = llama_n_ubatch(lctx);

    GGML_ASSERT(n_past + n_ubatch/2 + n_ubatch <= llama_n_ctx(lctx));

    llama_set_causal_attn(lctx, false);

    common_batch batch(lctx);

    bool ok = true;
    uint32_t pos = n_past;
    for (const uint32_t n_tokens : { n_ubatch/2, n_ubatch }) {
        batch.clear();
        for (uint32_t i = 0; i < n_tokens; i++) {
            batch.add(tokens[i], pos++, 0, true);
        }

        const int32_t rc = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (rc != 0) {
            LOG_ERR("%s: n_tokens=%u: llama_process returned %d\n", __func__, n_tokens, rc);
            ok = false;
            break;
        }

        const float * logits = llama_get_logits_ith(lctx, n_tokens - 1);
        if (logits == nullptr) {
            LOG_ERR("%s: n_tokens=%u: no logits\n", __func__, n_tokens);
            ok = false;
            break;
        }
        for (uint32_t j = 0; j < n_vocab; j++) {
            if (std::isnan(logits[j])) {
                LOG_ERR("%s: n_tokens=%u: nan logit\n", __func__, n_tokens);
                ok = false;
                break;
            }
        }
    }

    return ok;
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_FLINT:
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_MAION_CODER:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_CLOVER_MOE:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_DEEPSEEK41:
        case LLM_ARCH_XING4_0:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_BEAM:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_HY_V4:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_KIMI_K31:
        case LLM_ARCH_GLM5_NEXT:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MISTRAL_LARGE4:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_LAGUNA:
        case LLM_ARCH_MAPLE:
        case LLM_ARCH_MUSE_SPARK:
        case LLM_ARCH_KOLIBRI1:
        case LLM_ARCH_METIS16:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
        case LLM_ARCH_K2_HORIZON:
            return true;
        default:
            return false;
    }
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GRANITE_SWITCH) {
        return false; // FIXME adapter fixture
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_GEMMA_EMBEDDING2 || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_CLEF || arch == LLM_ARCH_VLLM_DECISION2) {
        return false; // TODO decision head tensors
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    // FIXME: these hit scheduler/view-backed-output issues with WebGPU on CI.
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_QWEN4EXP ||
            arch == LLM_ARCH_HY_V4) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    // FIXME: jamba produces incorrect output (~0.55 NMSE vs CPU) on the HIP
    // backend on RDNA3.5 (gfx1151); the SSM kernels need investigation.
#ifdef GGML_USE_HIP
    if (arch == LLM_ARCH_JAMBA) {
        return false;
    }
#endif // GGML_USE_HIP

    return true;
}

static int save_models(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity, const std::string & dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_arch_supports_save(arch) || !arch_supported(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llama_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            llama_model_ptr model = get_model(gguf_ctx.get(), nullptr, seed, stdev, {});
            const std::string path = dir + "/" + llama_arch_name(arch) + (moe ? "-moe.gguf" : "-dense.gguf");
            LOG_INF("%s: '%s %s' -> %s ...", __func__, llama_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model.get(), path.c_str());
            const std::uintmax_t size = std::filesystem::file_size(path);
            LOG_CNT(" (%.2f MB)\n", (double) size / (1024.0 * 1024.0));
        }
    }
    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return 0;
}

// the CPU and the GPU of an Apple SoC carry the same description, the name tells them apart (CPU, MTL0)
static std::string get_device_label(ggml_backend_dev_t dev) {
    const char * desc = ggml_backend_dev_description(dev);
    const char * name = ggml_backend_dev_name(dev);

    if (strcmp(desc, name) == 0) {
        return desc;
    }

    return std::string(desc) + " (" + name + ")";
}

static int test_backends(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity, const char * target_backend) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;
        bool                            host_experts; // keep the experts in host memory, see host_experts_test
        size_t                          moe_cache_size;
        std::vector<size_t>             moe_cache_sizes;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode, bool host_experts = false, size_t moe_cache_size = 0, std::vector<size_t> moe_cache_sizes = {})
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode), host_experts(host_experts), moe_cache_size(moe_cache_size), moe_cache_sizes(std::move(moe_cache_sizes)) {
            if (!this->moe_cache_sizes.empty()) {
                this->moe_cache_sizes.resize(llama_max_devices(), 0);
            }
        }
    };

    const llama_model_tensor_buft_override host_experts_overrides[] = {
        { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (target_backend != nullptr && strcmp(target_backend, ggml_backend_dev_name(dev)) != 0) {
                    continue;
                }
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, get_device_label(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }

        if (target_backend == nullptr) {
            dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);
        }

        // the ops that use the experts are offloaded to the first device and the scheduler copies the used experts
        if (!devices_meta.empty()) {
            dev_configs.emplace_back(devices_meta, "Host experts", LLAMA_SPLIT_MODE_LAYER, true);
            max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
        }

        // the ops that use the host experts run on a GPU and read the experts from a cache
        // the cache has only a few slots (4 for 288 KiB experts), so the experts are evicted and uploaded again
        if (!devices_meta.empty()) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(devices_meta[0]);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{devices_meta[0]}, "MoE cache", LLAMA_SPLIT_MODE_LAYER, true, 1536*1024);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{devices_meta[0]}, "MoE cache, explicit", LLAMA_SPLIT_MODE_LAYER, true, 0, std::vector<size_t>{1536*1024});
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{devices_meta[0]}, "MoE cache, too small", LLAMA_SPLIT_MODE_LAYER, true, 0, std::vector<size_t>{1});
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                // each GPU caches the layers assigned to it
                if (devices_meta.size() > 1) {
                    dev_configs.emplace_back(devices_meta, "MoE cache, layer split", LLAMA_SPLIT_MODE_LAYER, true, 1536*1024);
                    max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                    dev_configs.emplace_back(devices_meta, "MoE cache, unequal", LLAMA_SPLIT_MODE_LAYER, true, 0, std::vector<size_t>{1536*1024, 3*1024*1024});
                    max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                    std::reverse(devices_meta.begin(), devices_meta.end());
                    dev_configs.emplace_back(devices_meta, "MoE cache, reversed/zero", LLAMA_SPLIT_MODE_LAYER, true, 0, std::vector<size_t>{0, 1536*1024});
                    max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                }
            }
        }
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llama_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llama_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|%15s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|%15s %10s|\n";

    bool all_ok = true;
    size_t n_tests = 0;
    size_t n_failed = 0;
    common_log_flush(common_log_main());
    LOG(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip", "Mixed batch");
    LOG("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        LOG("-");
    }
    LOG("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        LOG("-");
    }
    LOG("|------|---------------|---------|---------------|\n");
    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue;
        }

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string config_name = moe ? "MoE" : "Dense";
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }
            std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_cpu;
            std::vector<float> logits_cpu;
            for (device_config & dc : dev_configs) {
                if (dc.host_experts && (!moe || !host_experts_test(arch))) {
                    continue;
                }
                const llama_model_tensor_buft_override * overrides = dc.host_experts ? host_experts_overrides : nullptr;

                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                LOG(template_row_cfg.c_str(), llama_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                std::string status_mixed     = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};
                char mixed_str[12] = {0};

                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty());
                bool test_executed = false;
                bool test_ok = true;
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                    }
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llama_arch_supports_sm_tensor(arch)) {
                        test_executed = true;
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, dc.devs, dc.split_mode, encode, overrides, dc.moe_cache_size, dc.moe_cache_sizes);
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        const double nmse_val = nmse(logits_cpu, logits_dev);
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (nmse_val > 1e-4) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }

                        // chunked decode matches a single batch only with causal attention over a memory
                        llama_context * lctx_dev = model_and_ctx_dev.second.get();
                        if (!encode && llama_get_memory(lctx_dev) != nullptr) {
                            std::vector<float> embd_mixed((size_t) llama_model_n_embd_inp(model_and_ctx_dev.first.get())*tokens.size()/4);
                            std::mt19937 gen(seed);
                            std::normal_distribution<float> dis(0.0f, stdev);
                            for (float & v : embd_mixed) {
                                v = dis(gen);
                            }
                            std::vector<float> logits_mixed;
                            std::vector<float> logits_chunks;
                            if (llama_arch_supports_mixed_batch(arch)) {
                                if (get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, false, logits_chunks) != 0 ||
                                    get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, true,  logits_mixed)  != 0) {
                                    throw std::runtime_error("failed to decode mixed batch");
                                }
                                const double nmse_mixed = nmse(logits_chunks, logits_mixed);
                                snprintf(mixed_str, sizeof(mixed_str), "(%.2e)", nmse_mixed);
                                status_mixed = "\033[1;32mOK\033[0m";
                                if (nmse_mixed > 1e-4) {
                                    test_ok = false;
                                    status_mixed = "\033[1;31mFAIL\033[0m";
                                }
                            } else {
                                // must be rejected as an invalid batch, mute the expected error log
                                ud.verbosity = LOG_LEVEL_OUTPUT;
                                const int32_t err = get_logits_mixed(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, embd_mixed, true, logits_mixed);
                                ud.verbosity = verbosity;
                                if (err != -1) {
                                    test_ok = false;
                                    status_mixed = "\033[1;31mFAIL\033[0m";
                                }
                            }
                        }

                        // runs after the mixed batch check, as it leaves the context with non-causal attention
                        if (!encode && !check_causal_attn_toggle(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens)) {
                            if (test_ok) {
                                status_nmse = "\033[1;31mFAIL\033[0m (toggle)";
                            }
                            test_ok = false;
                        }
                    }

                    if (llama_arch_supports_save(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        test_executed = true;
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                        //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                        const std::string path = get_tmp_gguf_path();
                        llama_model_save_to_file(model_and_ctx_dev.first.get(), path.c_str());
                        FILE * file = fopen(path.c_str(), "rb");
                        GGML_ASSERT(file != nullptr);

                        {
                            auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, stdev, dc.devs, dc.split_mode, encode, overrides, dc.moe_cache_size, dc.moe_cache_sizes);
                            const std::vector<float> logits_roundtrip = get_logits(
                                model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
                            status_roundtrip = "\033[1;32mOK\033[0m";
                            GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                            for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                                if (logits_roundtrip[i] != logits_dev[i]) {
                                    test_ok = false;
                                    status_roundtrip = "\033[1;31mFAIL\033[0m";
                                    break;
                                }
                            }
                        }

                        fclose(file);
                        std::filesystem::remove(path);
                    }
                }

                if (test_executed) {
                    n_tests++;
                    if (!test_ok) {
                        n_failed++;
                        all_ok = false;
                    }
                }

                // log the results for this test case
                LOG(template_row_res.c_str(), status_nmse.c_str(), nmse_str, status_roundtrip.c_str(), status_mixed.c_str(), mixed_str);
            }
        }
    }

    if (n_tests == 0) {
        LOG("Summary: no tests executed\n");
    } else if (n_failed == 0) {
        LOG("Summary: all %zu test(s) passed\n", n_tests);
    } else {
        LOG("Summary: %zu test(s) executed, %zu failed\n", n_tests, n_failed);
    }

    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return all_ok ? 0 : 1;
}

#ifdef LLAMA_TEST_FORK_INTERNALS
struct recovery_log {
    ggml_log_callback callback;
    void * user_data;
    int reserves = 0;
    int position_errors = 0;
    unsigned last_reserve_n_seqs = 0;

    recovery_log() {
        llama_log_get(&callback, &user_data);
        llama_log_set([](ggml_log_level level, const char * text, void * data) {
            auto & log = *static_cast<recovery_log *>(data);
            if (strstr(text, "non-consecutive token position") || strstr(text, "inconsistent sequence positions")) {
                ++log.position_errors;
            }
            if (strstr(text, "graph_reserve: reserving a graph for ubatch")) {
                ++log.reserves;
                const auto * n_seqs = strstr(text, "n_seqs = ");
                GGML_ASSERT(n_seqs && sscanf(n_seqs, "n_seqs = %u", &log.last_reserve_n_seqs) == 1);
            }
            if (level >= GGML_LOG_LEVEL_ERROR) {
                log.callback(level, text, log.user_data);
            }
        }, this);
    }

    ~recovery_log() {
        llama_log_set(callback, user_data);
    }
};

static std::vector<float> recovery_decode(llama_context * ctx, llama_pos & pos, int tokens, int outputs, bool raw, bool mtp, bool mixed = false) {
    const auto * model = llama_get_model(ctx);
    const size_t n_embd = llama_model_n_embd_inp(model);
    const size_t n_state = llama_model_n_embd(model);
    const size_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> embd(n_embd, 0.01f);
    std::vector<float> state(n_state, 0.02f);
    common_batch batch(ctx);
    for (int i = 0; i < tokens; ++i) {
        llama_pos p[4] = { pos, pos, pos, 0 };
        const bool output = i >= tokens - outputs;
        const int idx = raw && (!mixed || i % 2 == 0)
            ? batch.add_embd({ embd.data(), 1, n_embd }, p, 0, output)
            : batch.add(pos % n_vocab, pos, 0, output);
        if (mtp) {
            GGML_ASSERT(batch.set_embd_state(idx, { state.data(), 1, n_state }));
        }
        ++pos;
    }
    GGML_ASSERT(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
    llama_synchronize(ctx);
    std::vector<float> logits;
    for (int i = 0; i < outputs; ++i) {
        const float * row = llama_get_logits_ith(ctx, tokens - outputs + i);
        GGML_ASSERT(row);
        for (size_t j = 0; j < n_vocab; ++j) {
            GGML_ASSERT(std::isfinite(row[j]));
            logits.push_back(row[j]);
        }
    }
    return logits;
}

static void test_text_plan_retention(size_t seed, float stdev, recovery_log & log) {
    auto meta = get_gguf_ctx(LLM_ARCH_LLAMA, false);
    auto mp = llama_model_default_params();
    // Two CPU backends make allocation failures use the scheduler callback, even with aliased buffer types.
    ggml_backend_dev_t devices[] = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr };
    GGML_ASSERT(devices[0]);
    mp.devices = devices;
    mp.n_gpu_layers = 0;
    tensor_data_params data = { seed, stdev };
    llama_model_ptr model(llama_model_init_from_user(meta.get(), set_tensor_data, &data, mp));
    GGML_ASSERT(model);
    auto cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = cp.n_ubatch = 64;
    cp.n_outputs_max = cp.n_outputs_max_per_seq = 49;
    cp.n_threads = cp.n_threads_batch = 2;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
    llama_context_ptr reference(llama_init_from_model(model.get(), cp));
    GGML_ASSERT(ctx && reference);
    llama_pos pos = 0;
    llama_pos ref_pos = 0;
    recovery_decode(ctx.get(), pos, 4, 1, false, false);
    recovery_decode(reference.get(), ref_pos, 4, 1, false, false);
    const auto scheduler = ctx->get_sched();
    GGML_ASSERT(ggml_backend_sched_get_n_backends(scheduler) == 2);

    // Emulate a text reservation that does not cover the next valid batch.
    auto mctx = ctx->get_memory()->init_full();
    GGML_ASSERT(ctx->graph_reserve(1, 1, 1, mctx.get()));
    auto perf = llama_perf_context(ctx.get());
    const auto reallocs_before = perf.n_reallocs_same + perf.n_reallocs_diff;
    for (int step = 0; step < 8; ++step) {
        const int reserves_before = log.reserves;
        auto actual = recovery_decode(ctx.get(), pos, 4, 4, false, false);
        GGML_ASSERT(log.reserves == reserves_before);
        auto expected = recovery_decode(reference.get(), ref_pos, 4, 4, false, false);
        GGML_ASSERT(actual == expected);
        const auto current = llama_perf_context(ctx.get());
        if (step == 0) {
            GGML_ASSERT(current.n_reallocs_same + current.n_reallocs_diff > reallocs_before);
            perf = current;
        } else {
            GGML_ASSERT(current.n_reallocs_same == perf.n_reallocs_same && current.n_reallocs_diff == perf.n_reallocs_diff);
        }
        GGML_ASSERT(ctx->get_sched() == scheduler);
    }
    printf("graph recovery: undersized text plan retains runtime reservation passed\n");
}

static void test_memory_update_reservation(size_t seed, float stdev, recovery_log & log) {
    for (auto arch : { LLM_ARCH_KIMI_LINEAR, LLM_ARCH_MINIMAX_01 }) {
        auto meta = get_gguf_ctx(arch, true);
        auto mp = llama_model_default_params();
        ggml_backend_dev_t devices[] = { nullptr };
        mp.devices = devices;
        mp.n_gpu_layers = 0;
        tensor_data_params data = { seed, stdev };
        llama_model_ptr model(llama_model_init_from_user(meta.get(), set_tensor_data, &data, mp));
        GGML_ASSERT(model);
        auto cp = llama_context_default_params();
        cp.n_ctx = 256;
        cp.n_batch = cp.n_ubatch = 16;
        cp.n_seq_max = 2;
        cp.n_outputs_max = 2;
        cp.n_outputs_max_per_seq = 1;
        cp.n_threads = cp.n_threads_batch = 2;
        llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
        GGML_ASSERT(ctx && log.last_reserve_n_seqs == 1);
        llama_pos pos = 0;
        recovery_decode(ctx.get(), pos, 4, 0, true, false);
        auto * memory = ctx->get_memory();
        GGML_ASSERT(memory->get_can_shift());
        memory->seq_add(0, 0, -1, 1);
        ++pos;
        const int before = log.reserves;
        GGML_ASSERT(ctx->memory_update(false));
        llama_synchronize(ctx.get());
        GGML_ASSERT(log.reserves > before && log.last_reserve_n_seqs == cp.n_seq_max);
        const int after = log.reserves;
        recovery_decode(ctx.get(), pos, 1, 1, false, false);
        GGML_ASSERT(log.reserves == after);
        printf("graph recovery: %s memory update preserves n_seq_max passed\n", llm_arch_name(arch));
    }
}

static void test_dflash_tail_equivalence(
        gguf_context * target_meta, gguf_context * draft_meta,
        llama_model_params mp, llama_context_params cp, tensor_data_params data) {
    // Separate random models keep the selector comparison independent of the constant-logit acceptance tests.
    llama_model_ptr target(llama_model_init_from_user(target_meta, set_tensor_data, &data, mp));
    llama_model_ptr draft(llama_model_init_from_user(draft_meta, set_tensor_data, &data, mp));
    GGML_ASSERT(target && draft);
    cp.cb_eval = nullptr;
    cp.cb_eval_user_data = nullptr;
    cp.ctx_other = nullptr;
    llama_context_ptr ctx_tgt(llama_init_from_model(target.get(), cp));
    cp.ctx_other = ctx_tgt.get();
    std::map<std::string, std::vector<float>> injected;
    cp.cb_eval_user_data = &injected;
    cp.cb_eval = [](ggml_tensor * t, bool ask, void * ud) {
        if (strncmp(t->name, "Kcur_injected", 13) && strncmp(t->name, "Vcur_injected", 13)) {
            return false;
        }
        if (ask) {
            return true;
        }
        GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
        auto & rows = (*static_cast<std::map<std::string, std::vector<float>> *>(ud))[t->name];
        const size_t offset = rows.size();
        rows.resize(offset + ggml_nelements(t));
        ggml_backend_tensor_get(t, rows.data() + offset, 0, ggml_nbytes(t));
        return true;
    };
    llama_context_ptr ctx_dft(llama_init_from_model(draft.get(), cp));
    GGML_ASSERT(ctx_tgt && ctx_dft);
    common_params_speculative params;
    params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH };
    params.draft.ctx_tgt = ctx_tgt.get();
    params.draft.ctx_dft = ctx_dft.get();
    params.draft.n_max = 3;
    params.draft.n_min = 1;
    common_speculative_ptr spec(common_speculative_init(params, 2));
    GGML_ASSERT(spec);
    const bool mrope = llama_model_rope_type(draft.get()) == LLAMA_ROPE_TYPE_MROPE;
    const auto kv_type = cp.type_k;
    auto process = [&](const common_batch & in) { return common_speculative_process(spec.get(), in); };
    for (int media : { 0, 1, 2 }) {
        std::vector<float> full_lattice;
        std::map<std::string, std::vector<float>> full_injected;
        llama_tokens full_result;
        for (bool tail : { false, true }) {
            llama_memory_clear(llama_get_memory(ctx_tgt.get()), true);
            llama_memory_clear(llama_get_memory(ctx_dft.get()), true);
            injected.clear();
            const llama_pos end = media == 0 ? 128 : media == 1 ? 131 : 134;
            common_speculative_set_prompt_end(spec.get(), 0, tail ? end : 0);
            llama_tokens prompt;
            for (int pos = 0; pos < 128;) {
                common_batch in(ctx_tgt.get());
                for (int i = 0; i < 16; ++i) {
                    in.add(1, pos++, 0, true);
                    prompt.push_back(1);
                }
                GGML_ASSERT(llama_process(ctx_tgt.get(), LLAMA_PROCESS_TYPE_DECODE, in.get()) == 0);
                GGML_ASSERT(process(in));
            }
            if (media) {
                common_batch image(ctx_tgt.get());
                std::vector<float> embeddings(6*256, 0.01f);
                for (int i = 0; i < 6; ++i) {
                    const llama_pos p[4] = { 128, 128 + i/3, 128 + i%3, 0 };
                    image.add_embd({ embeddings.data() + i*256, 1, 256 }, p, 0, true);
                    prompt.push_back(0);
                }
                GGML_ASSERT(llama_process(ctx_tgt.get(), LLAMA_PROCESS_TYPE_DECODE, image.get()) == 0);
                GGML_ASSERT(process(image));
                if (media == 2) {
                    common_batch text(ctx_tgt.get());
                    for (int i = 131; i < end; ++i) {
                        text.add(1, i, 0, true);
                        prompt.push_back(1);
                    }
                    GGML_ASSERT(llama_process(ctx_tgt.get(), LLAMA_PROCESS_TYPE_DECODE, text.get()) == 0);
                    GGML_ASSERT(process(text));
                }
            }
            common_speculative_begin(spec.get(), 0, prompt);
            llama_tokens result;
            common_speculative_get_draft_params(spec.get(), 1).drafting = false;
            auto & dp = common_speculative_get_draft_params(spec.get(), 0);
            dp.drafting = true;
            dp.pos0 = end;
            dp.id_last = 0;
            dp.prompt = &prompt;
            dp.result = &result;
            common_speculative_draft(spec.get());
            const float * lattice = llama_get_embeddings_nextn(ctx_dft.get());
            GGML_ASSERT(lattice && result.size() == 3);
            std::vector<float> actual(lattice, lattice + 4*256);
            if (tail) {
                GGML_ASSERT(result == full_result);
                for (size_t i = 0; i < actual.size(); ++i) {
                    GGML_ASSERT(std::isfinite(actual[i]) && std::abs(full_lattice[i] - actual[i]) < 1e-5f);
                }
                GGML_ASSERT(!injected.empty() && full_injected.size() == injected.size());
                for (const auto & entry : injected) {
                    const auto & rows = entry.second;
                    const auto & full = full_injected.at(entry.first);
                    GGML_ASSERT(!rows.empty() && rows.size() < full.size());
                    GGML_ASSERT(std::any_of(rows.begin(), rows.end(), [](float v) { return std::abs(v) > 1e-6f; }));
                    const size_t offset = full.size() - rows.size();
                    for (size_t i = 0; i < rows.size(); ++i) {
                        GGML_ASSERT(std::isfinite(rows[i]) && std::abs(full[offset + i] - rows[i]) < 1e-5f);
                    }
                }
            } else {
                full_lattice = std::move(actual);
                full_injected = injected;
                full_result = result;
            }
        }
        printf("DFlash2 full/tail K/V and selector equivalence, media=%d, draft-rope=%s, KV=%s, tensors=%zu\n", media, mrope ? "mrope" : "1d", ggml_type_name(kv_type), injected.size());
    }
}

static void test_dflash_reservation(llama_model * model, llama_model * measure_model, llama_context_params cp) {
    cp.n_seq_max = 1;
    cp.n_batch = cp.n_ubatch = 1024;
    cp.n_ctx = 4096;
    cp.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    llama_context_ptr compact(llama_init_from_model(model, cp));
    llama_context_ptr reference(llama_init_from_model(model, cp));
    llama_context_ptr measured(llama_init_from_model(measure_model, cp));
    GGML_ASSERT(compact && reference && measured);
    const auto compute_bytes = [](llama_context * ctx) {
        size_t total = 0;
        for (const auto & entry : ctx->memory_breakdown()) {
            total += entry.second.compute;
        }
        return total;
    };
    const size_t compact_bytes = compute_bytes(compact.get());
    const auto estimate = measured->memory_breakdown();
    for (const auto & entry : compact->memory_breakdown()) {
        const auto it = estimate.find(entry.first);
        GGML_ASSERT(it != estimate.end() && it->second.compute >= entry.second.compute);
    }
    // Fit must cover both external sources and the conservative source-storage budget.
    const auto sched = measured->get_sched();
    std::vector<size_t> sizes(ggml_backend_sched_get_n_backends(sched));
    auto measure_mctx = measured->get_memory()->init_full();
    for (auto input : { llama_dflash_input::DEVICE_OWNED, llama_dflash_input::DEVICE }) {
        GGML_ASSERT(measured->graph_reserve(cp.n_ubatch, 1, 1, measure_mctx.get(), true, sizes.data(),
                measured->kq_mask_packed_reachable(), input));
        for (size_t i = 0; i < sizes.size(); ++i) {
            const auto backend = ggml_backend_sched_get_backend(sched, i);
            const auto buft = ggml_backend_sched_get_buffer_type(sched, backend);
            const auto it = estimate.find(buft);
            GGML_ASSERT(it != estimate.end() && it->second.compute >= sizes[i]);
        }
    }
    GGML_ASSERT(compact->kq_mask_packed_reachable());
    GGML_ASSERT(compact->get_gf_res_reserve()->get_logits()->ne[1] == 4);
    auto mctx = reference->get_memory()->init_full();
    GGML_ASSERT(reference->graph_reserve(cp.n_ubatch, 1, cp.n_ubatch, mctx.get(), false, nullptr,
            reference->kq_mask_packed_reachable()));
    const size_t reference_bytes = compute_bytes(reference.get());
    GGML_ASSERT(compact_bytes < reference_bytes);

    int pos = 0;
    int bound = 4;
    const int width = model->hparams.n_embd_inp_enc();
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int ingest_rows[] = { (int) cp.n_ubatch - 1, (int) cp.n_ubatch, 12 };
    int step = 0;
    for (int tokens : { 4, 8, 4 }) {
        const int count = ingest_rows[step++];
        bound = std::max(bound, tokens);
        std::vector<float> logits[2];
        std::vector<uint8_t> state[2];
        int which = 0;
        for (auto * ctx : { compact.get(), reference.get() }) {
            common_batch ingest(ctx);
            std::vector<float> features(count * width, 0.01f);
            for (int i = 0; i < count; ++i) {
                const llama_pos p[4] = { pos + i, pos + i, pos + i, 0 };
                ingest.add_embd({ features.data() + i * width, 1, (size_t) width }, p, 0, false);
            }
            GGML_ASSERT(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, ingest.get()) == 0);
            common_batch noise(ctx);
            for (int i = 0; i < tokens; ++i) {
                noise.add(i == 0 ? 1 : 127, pos + count + i, 0, true);
            }
            GGML_ASSERT(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, noise.get()) == 0);
            for (int i = 0; i < tokens; ++i) {
                const float * row = llama_get_logits_ith(ctx, i);
                GGML_ASSERT(row);
                for (int j = 0; j < vocab; ++j) {
                    GGML_ASSERT(std::isfinite(row[j]));
                    logits[which].push_back(row[j]);
                }
            }
            GGML_ASSERT(ctx->get_gf_res_reserve()->get_logits()->ne[1] == bound);
            GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx), 0, pos + count, -1));
            state[which].resize(llama_state_seq_get_size(ctx, 0));
            GGML_ASSERT(llama_state_seq_get_data(ctx, state[which].data(), state[which].size(), 0) == state[which].size());
            ++which;
        }
        GGML_ASSERT(logits[0] == logits[1] && state[0] == state[1]);
        if (tokens == 4 && bound == 4) {
            GGML_ASSERT(compute_bytes(compact.get()) == compact_bytes);
        }
        pos += count;
    }
    printf("DFlash2 reserve KV=%s: compact=%zu full-noise=%zu bytes; F1=1023/1024, F2 logits, rollback and larger-noise fallback passed\n",
            ggml_type_name(cp.type_k), compact_bytes, reference_bytes);
}

// Exercise device staging/gathers on the CPU backend as a numerical oracle. This
// deliberately calls the internal staging method: automatic CPU use stays off.
static int test_dflash_device_inputs(size_t seed, float stdev, const char * backend_name, int n_seqs) {
    auto target_meta = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true, 0, 4096);
    auto draft_meta = get_gguf_ctx(LLM_ARCH_DFLASH, false, 0, 4096);
    // Five feature streams keep the target small while exercising F1 concatenation.
    const int32_t layers[] = { 0, 1, 0, 1, 1 };
    constexpr int n_features = sizeof(layers) / sizeof(layers[0]);
    gguf_set_arr_data(draft_meta.get(), "dflash.target_layers", GGUF_TYPE_INT32, layers, n_features);
    gguf_set_val_u32(draft_meta.get(), "dflash.block_size", 4);
    gguf_set_val_u32(draft_meta.get(), "dflash.conv_kernel_size", 3);
    gguf_set_val_u32(draft_meta.get(), "dflash.conv_group_size", 8);
    gguf_set_val_u32(draft_meta.get(), "dflash.selector_rank", 8);
    gguf_set_val_u32(draft_meta.get(), "dflash.selector_top_k", 2);
    gguf_set_val_u32(draft_meta.get(), "tokenizer.ggml.mask_token_id", 127);
    const uint32_t sections[] = { 64, 0, 0, 0 };
    gguf_set_arr_data(draft_meta.get(), "dflash.rope.dimension_sections", GGUF_TYPE_UINT32, sections, 4);
    const bool swa[] = { true, true };
    gguf_set_arr_data(draft_meta.get(), "dflash.attention.sliding_window_pattern", GGUF_TYPE_BOOL, swa, 2);
    auto mp = llama_model_default_params();
    const auto dev = backend_name == nullptr ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU) :
                                              ggml_backend_dev_by_name(backend_name);
    GGML_ASSERT(dev);
    const bool gpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU;
    if (gpu && n_seqs != 1) {
        return 0;
    }
    ggml_backend_dev_t devices[] = { gpu ? dev : nullptr, nullptr };
    mp.devices = devices;
    mp.n_gpu_layers = gpu ? 99 : 0;
    tensor_data_params data = { seed, stdev };
    llama_model_ptr target(llama_model_init_from_user(target_meta.get(), set_tensor_data, &data, mp));
    llama_model_ptr draft(llama_model_init_from_user(draft_meta.get(), set_tensor_data, &data, mp));
    GGML_ASSERT(target && draft);
    auto measure_mp = mp;
    measure_mp.no_alloc = true;
    llama_model_ptr measure_draft(llama_model_init_from_user(draft_meta.get(), set_tensor_data, &data, measure_mp));
    GGML_ASSERT(measure_draft);
    const int width = llama_model_n_embd(target.get());

    bool reserve_ok = true;
    for (auto kv : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
        auto cp = llama_context_default_params();
        cp.n_ctx = 256;
        cp.n_batch = 32;
        cp.n_ubatch = 4;
        cp.n_seq_max = n_seqs;
        cp.n_threads = cp.n_threads_batch = 2;
        cp.type_k = cp.type_v = kv;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_context_ptr tgt(llama_init_from_model(target.get(), cp));
        GGML_ASSERT(tgt);
        for (int lid : layers) {
            llama_set_embeddings_layer_inp(tgt.get(), lid, true);
        }
        GGML_ASSERT(tgt->prepare_layer_inputs_device(dev));
        cp.n_ubatch = 16;
        cp.ctx_other = tgt.get();
        if (n_seqs == 1) {
            test_dflash_reservation(draft.get(), measure_draft.get(), cp);
        }
        llama_context_ptr host(llama_init_from_model(draft.get(), cp));
        llama_context_ptr device(llama_init_from_model(draft.get(), cp));
        GGML_ASSERT(host && device);
        const char * reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        const bool native_inputs = gpu && (strcmp(reg, "ROCm") == 0 || strcmp(reg, "CUDA") == 0);
        GGML_ASSERT(llama_dflash_init_device_inputs(device.get(), tgt.get()) == native_inputs);
        for (auto * ctx : { host.get(), device.get() }) {
            llama_set_causal_attn(ctx, false);
            llama_set_embeddings_nextn(ctx, true, false);
            ctx->sched_reserve();
        }
        const auto device_reserve = device->memory_breakdown();
        auto previous_compute = device_reserve;
        const auto check_reserve = [&](const char * phase, int count, int offset) {
            if (n_seqs != 1) {
                return;
            }
            for (const auto & entry : device->memory_breakdown()) {
                const auto initial = device_reserve.find(entry.first);
                const auto previous = previous_compute.find(entry.first);
                if (initial == device_reserve.end() || entry.second.compute > initial->second.compute) {
                    reserve_ok = false;
                    if (previous == previous_compute.end() || entry.second.compute > previous->second.compute) {
                        fprintf(stderr, "DFlash2 reserve growth: backend=%s KV=%s phase=%s tokens=%d offset=%d buft=%s initial=%zu previous=%zu current=%zu bytes\n",
                                ggml_backend_dev_name(dev), ggml_type_name(kv), phase, count, offset,
                                ggml_backend_buft_name(entry.first), initial == device_reserve.end() ? 0 : initial->second.compute,
                                previous == previous_compute.end() ? 0 : previous->second.compute, entry.second.compute);
                    }
                }
                previous_compute[entry.first].compute = entry.second.compute;
            }
        };
        int pos = 0;
        for (int count : { n_seqs == 1 ? 31 : 15, 1, 4 }) {
            common_batch in(tgt.get());
            for (int i = 0; i < count; ++i) {
                // Interleaving forces the recurrent target to reorder rows across ubatches.
                for (int seq = 0; seq < n_seqs; ++seq) {
                    in.add(1 + i, pos + i, seq, true);
                }
            }
            GGML_ASSERT(llama_process(tgt.get(), LLAMA_PROCESS_TYPE_DECODE, in.get()) == 0);
            // Logits access must not destroy the independent device-row mapping.
            GGML_ASSERT(llama_get_logits_ith(tgt.get(), 0));
            const auto & rows = llama_dflash_device_rows(tgt.get());
            GGML_ASSERT(rows.size() == (size_t) in.size());
            common_batch dev_in(device.get());
            common_batch host_in(host.get());
            std::vector<float> features((size_t) in.size()*width*n_features);
            for (int k = 0; k < n_features; ++k) {
                const float * extracted = llama_get_embeddings_layer_inp(tgt.get(), layers[k]);
                std::vector<float> raw((size_t) in.size()*width);
                ggml_backend_tensor_get(tgt->get_layer_input_device(layers[k]), raw.data(), 0, raw.size()*sizeof(float));
                for (int i = 0; i < in.size(); ++i) {
                    GGML_ASSERT(memcmp(extracted + (size_t) i*width, raw.data() + (size_t) rows[i]*width, width*sizeof(float)) == 0);
                    std::memcpy(features.data() + (size_t) i*width*n_features + k*width,
                            extracted + (size_t) i*width, width*sizeof(float));
                }
            }
            for (int i = 0; i < in.size(); ++i) {
                const auto & tok = in.tokens[i];
                dev_in.add(rows[i], tok.pos[0], tok.seq_id, false);
                const llama_pos p[4] = { tok.pos[0], tok.pos[0], tok.pos[0], 0 };
                host_in.add_embd({ features.data() + (size_t) i*width*n_features, 1, (size_t) width*n_features }, p, tok.seq_id, false);
            }
            // Non-causal F1 calls must fit in one ubatch, as in the speculative caller.
            const int32_t n_ubatch = (int32_t) llama_n_ubatch(device.get());
            for (int32_t offset = 0; offset < in.size(); offset += n_ubatch) {
                const int32_t n_chunk = std::min(n_ubatch, in.size() - offset);
                GGML_ASSERT(llama_dflash_inject_device(device.get(), dev_in.get_sub_batch(offset, n_chunk)) == 0);
                GGML_ASSERT(!device->get_cparams().dflash_device_inputs);
                check_reserve("F1-device", n_chunk, offset);
                GGML_ASSERT(llama_process(host.get(), LLAMA_PROCESS_TYPE_DECODE, host_in.get_sub_batch(offset, n_chunk)) == 0);
            }
            pos += count;
            for (int seq = 0; seq < n_seqs; ++seq) {
                const size_t bytes = llama_state_seq_get_size(host.get(), seq);
                std::vector<uint8_t> a(bytes), b(bytes);
                GGML_ASSERT(bytes && llama_state_seq_get_size(device.get(), seq) == bytes);
                GGML_ASSERT(llama_state_seq_get_data(host.get(), a.data(), bytes, seq) == bytes);
                GGML_ASSERT(llama_state_seq_get_data(device.get(), b.data(), bytes, seq) == bytes);
                GGML_ASSERT(a == b);
            }
            check_reserve("state-readback", count*n_seqs, 0);
            // Alternate row-index injections and noise token batches of the same width.
            std::vector<float> lattice[2];
            int which = 0;
            for (auto * ctx : { host.get(), device.get() }) {
                common_batch noise(ctx);
                for (int seq = 0; seq < n_seqs; ++seq) {
                    for (int i = 0; i < 4; ++i) {
                        noise.add(i == 0 ? 1 : 127, pos + i, seq, false);
                    }
                }
                GGML_ASSERT(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, noise.get()) == 0);
                if (ctx == device.get()) {
                    check_reserve("F2-noise", noise.size(), 0);
                }
                const auto * embd = llama_get_embeddings_nextn(ctx);
                GGML_ASSERT(embd);
                lattice[which++].assign(embd, embd + n_seqs*4*llama_model_n_embd_out(draft.get()));
                for (int seq = 0; seq < n_seqs; ++seq) {
                    GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx), seq, pos, -1));
                }
            }
            GGML_ASSERT(lattice[0] == lattice[1]);
            check_reserve("rollback", count*n_seqs, 0);
        }
        common_batch invalid(device.get());
        invalid.add(31, pos, 0, false);
        GGML_ASSERT(llama_dflash_inject_device(device.get(), invalid.get()) == -1);
        GGML_ASSERT(!device->get_cparams().dflash_device_inputs);
        printf("DFlash2 device/host inputs: numerical checks passed, KV=%s n_seqs=%d\n", ggml_type_name(kv), n_seqs);
    }
    if (!reserve_ok) {
        fprintf(stderr, "DFlash2 reserve check failed: compute buffers grew beyond their initial reservation\n");
    }
    return reserve_ok ? 0 : 1;
}

static int test_dflash_selector(size_t seed, float stdev, const char * backend_name) {
    auto * head = backend_name ? ggml_backend_dev_by_name(backend_name) : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(head);
    const bool gpu = ggml_backend_dev_type(head) == GGML_BACKEND_DEVICE_TYPE_GPU;
    auto * peer = head;
    if (gpu) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * dev = ggml_backend_dev_get(i);
            if (dev != head && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                    ggml_backend_dev_backend_reg(dev) == ggml_backend_dev_backend_reg(head)) {
                peer = dev;
                break;
            }
        }
        if (peer == head) {
            fprintf(stderr, "DFlash2 selector: two GPU devices are required\n");
            return 1;
        }
    }
    constexpr int vocab = 4096;
    auto target_meta = get_gguf_ctx(LLM_ARCH_LLAMA, false, 0, vocab);
    auto draft_meta = get_gguf_ctx(LLM_ARCH_DFLASH, false, 0, vocab);
    const int32_t layers[] = { 0, 1 };
    gguf_set_arr_data(draft_meta.get(), "dflash.target_layers", GGUF_TYPE_INT32, layers, 2);
    gguf_set_val_u32(draft_meta.get(), "dflash.block_size", 4);
    gguf_set_val_u32(draft_meta.get(), "dflash.conv_kernel_size", 2);
    gguf_set_val_u32(draft_meta.get(), "dflash.conv_group_size", 8);
    gguf_set_val_u32(draft_meta.get(), "dflash.selector_rank", 32);
    gguf_set_val_u32(draft_meta.get(), "dflash.selector_top_k", 2);
    const bool swa[] = { true, true };
    gguf_set_arr_data(draft_meta.get(), "dflash.attention.sliding_window_pattern", GGUF_TYPE_BOOL, swa, 2);
    ggml_backend_dev_t devices[] = { gpu ? head : nullptr, gpu ? peer : nullptr, nullptr };
    const llama_model_tensor_buft_override head_weights[] = { { ".*", ggml_backend_dev_buffer_type(head) }, { nullptr, nullptr } };
    const llama_model_tensor_buft_override peer_weights[] = { { ".*", ggml_backend_dev_buffer_type(peer) }, { nullptr, nullptr } };
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = gpu ? 99 : 0;
    mp.tensor_buft_overrides = head_weights;
    tensor_data_params data = { seed, stdev };
    llama_model_ptr target(llama_model_init_from_user(target_meta.get(), set_tensor_data, &data, mp));
    mp.tensor_buft_overrides = peer_weights;
    llama_model_ptr draft(llama_model_init_from_user(draft_meta.get(), set_tensor_data, &data, mp));
    GGML_ASSERT(target && draft);
    // Virtual models do not discover optional selector tensors through the file index.
    ggml_context_ptr selector_ctx(ggml_init({3*ggml_tensor_overhead(), nullptr, true}));
    draft->dflash_selector_hidden = ggml_new_tensor_2d(selector_ctx.get(), GGML_TYPE_Q8_0, 256, 32);
    draft->dflash_selector_prev = ggml_new_tensor_2d(selector_ctx.get(), GGML_TYPE_Q8_0, 32, vocab);
    draft->dflash_selector_next = ggml_new_tensor_2d(selector_ctx.get(), GGML_TYPE_Q8_0, 32, vocab);
    ggml_set_name(draft->dflash_selector_hidden, "selector_hidden.weight");
    ggml_set_name(draft->dflash_selector_prev, "selector_predecessor.weight");
    ggml_set_name(draft->dflash_selector_next, "selector_successor.weight");
    ggml_backend_buffer_ptr selector_buffer(ggml_backend_alloc_ctx_tensors_from_buft(selector_ctx.get(), ggml_backend_dev_buffer_type(peer)));
    GGML_ASSERT(selector_buffer);
    ggml_backend_buffer_set_usage(selector_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (auto * t = ggml_get_first_tensor(selector_ctx.get()); t; t = ggml_get_next_tensor(selector_ctx.get(), t)) {
        std::vector<float> values(ggml_nelements(t));
        std::vector<uint8_t> packed(ggml_nbytes(t));
        fill_normal(values, t->name, data);
        ggml_quantize_chunk(t->type, values.data(), packed.data(), 0, t->ne[1], t->ne[0], nullptr);
        ggml_backend_tensor_set(t, packed.data(), 0, packed.size());
    }
    draft->hparams.dflash_selector_rank = 32;
    draft->output = nullptr;
    draft->tok_embd = nullptr;
    if (gpu) {
        GGML_ASSERT(ggml_backend_buft_get_device(ggml_backend_buffer_get_type(target->output->buffer)) == head);
        GGML_ASSERT(ggml_backend_buft_get_device(ggml_backend_buffer_get_type(draft->dflash_selector_hidden->buffer)) == peer);
    }

    struct capture {
        llama_context * ctx = nullptr;
        ggml_backend_dev_t head = nullptr;
        bool placement = false;
        int calls = 0;
        int64_t values = 0;
    } cap;
    cap.head = head;
    cap.placement = gpu && strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(head)), "ROCm") == 0;
    for (const auto kv : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
        auto cp = llama_context_default_params();
        cp.n_ctx = 128;
        cp.n_batch = cp.n_ubatch = 16;
        cp.n_threads = cp.n_threads_batch = 2;
        cp.type_k = cp.type_v = kv;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_context_ptr tgt(llama_init_from_model(target.get(), cp));
        GGML_ASSERT(tgt);
        cp.ctx_other = tgt.get();
        cp.cb_eval_user_data = &cap;
        cp.cb_eval = [](ggml_tensor * t, bool ask, void * userdata) {
            const bool unary = t->op == GGML_OP_GET_ROWS && t->src[0]->type == GGML_TYPE_F32 &&
                t->src[0]->ne[0] == 1 && t->src[0]->ne[1] == vocab;
            if (ask || !unary) return unary;
            auto & cap = *(capture *) userdata;
            const auto backend = ggml_backend_sched_get_tensor_backend(cap.ctx->get_sched(), t);
            GGML_ASSERT(backend);
            if (cap.placement) GGML_ASSERT(ggml_backend_get_device(backend) == cap.head);
            std::vector<float> logits(ggml_nelements(t->src[0])), values(ggml_nelements(t));
            std::vector<int32_t> indices(ggml_nelements(t->src[1]));
            ggml_backend_tensor_get(t->src[0], logits.data(), 0, ggml_nbytes(t->src[0]));
            ggml_backend_tensor_get(t->src[1], indices.data(), 0, ggml_nbytes(t->src[1]));
            ggml_backend_tensor_get(t, values.data(), 0, ggml_nbytes(t));
            const int64_t top_k = t->src[1]->ne[0];
            GGML_ASSERT(indices.size() == values.size());
            for (size_t i = 0; i < values.size(); ++i) {
                GGML_ASSERT(indices[i] >= 0 && indices[i] < vocab);
                const float expected = logits[(i / top_k)*vocab + indices[i]];
                GGML_ASSERT(std::isfinite(values[i]) && values[i] == expected);
            }
            ++cap.calls;
            cap.values += values.size();
            return true;
        };
        llama_context_ptr ctx(llama_init_from_model(draft.get(), cp));
        GGML_ASSERT(ctx);
        cap.ctx = ctx.get();
        llama_set_causal_attn(ctx.get(), false);
        llama_set_embeddings_nextn(ctx.get(), true, false);
        std::vector<float> features(16*2*256);
        for (size_t i = 0; i < features.size(); ++i) features[i] = stdev * (int(i % 29) - 14) / 14;
        common_batch prefix(ctx.get());
        for (int i = 0; i < 16; ++i) {
            const llama_pos pos[] = { i, i, i, 0 };
            prefix.add_embd({ features.data() + i*512, 1, 512 }, pos, 0, false);
        }
        GGML_ASSERT(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, prefix.get()) == 0);
        const int before = cap.calls;
        for (int count : { 1, 2, 4 }) {
            common_batch noise(ctx.get());
            for (int i = 0; i < count; ++i) noise.add(i ? 127 : 1, 16 + i, 0, false);
            const int calls = cap.calls;
            GGML_ASSERT(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, noise.get()) == 0);
            GGML_ASSERT(llama_get_embeddings_nextn(ctx.get()));
            GGML_ASSERT(cap.calls == calls + (count > 1));
            GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, 16, -1));
        }
        GGML_ASSERT(cap.calls == before + 2);
        printf("DFlash2 selector KV=%s head=%s selector=%s: widths 1/2/4, exact score gathers and rollback passed\n",
                ggml_type_name(kv), ggml_backend_dev_name(head), ggml_backend_dev_name(peer));
    }
    GGML_ASSERT(cap.calls == 4 && cap.values == 24);
    return 0;
}

static int test_dflash_vision(size_t seed, float stdev, int tail_mode = -1) {
    recovery_log log;
    auto target_meta = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
    auto draft_meta = get_gguf_ctx(LLM_ARCH_DFLASH, false);
    const int32_t target_layers[] = { 0, 1 };
    gguf_set_arr_data(draft_meta.get(), "dflash.target_layers", GGUF_TYPE_INT32, target_layers, 2);
    gguf_set_val_u32(draft_meta.get(), "dflash.block_size", 4);
    gguf_set_val_u32(draft_meta.get(), "dflash.conv_kernel_size", 3);
    gguf_set_val_u32(draft_meta.get(), "dflash.conv_group_size", 8);
    gguf_set_val_u32(draft_meta.get(), "dflash.selector_rank", 8);
    gguf_set_val_u32(draft_meta.get(), "dflash.selector_top_k", 2);
    const bool swa_pattern[] = { true, true };
    gguf_set_arr_data(draft_meta.get(), "dflash.attention.sliding_window_pattern", GGUF_TYPE_BOOL, swa_pattern, 2);
    if (tail_mode == 1) {
        const bool mixed[] = { true, false };
        gguf_set_arr_data(draft_meta.get(), "dflash.attention.sliding_window_pattern", GGUF_TYPE_BOOL, mixed, 2);
    } else if (tail_mode == 2) {
        gguf_remove_key(draft_meta.get(), "dflash.attention.sliding_window");
    }
    const char * tail_env = getenv("LLAMA_DFLASH_INGEST_TAIL");
    const int window = tail_mode == 0 && (!tail_env || atoi(tail_env) != 0) ? 32 : 0;
    gguf_set_val_u32(draft_meta.get(), "tokenizer.ggml.mask_token_id", 127);
    auto add_tensor_meta = [](gguf_context * meta, const char * name, int64_t rows) {
        ggml_tensor t = {};
        t.type = GGML_TYPE_F32;
        t.ne[0] = 256;
        t.ne[1] = rows;
        t.ne[2] = t.ne[3] = 1;
        ggml_set_name(&t, name);
        gguf_add_tensor(meta, &t);
    };
    add_tensor_meta(target_meta.get(), "output.weight", 128);
    add_tensor_meta(draft_meta.get(), "output.weight", 128);
    add_tensor_meta(draft_meta.get(), "selector_hidden.weight", 8);
    tensor_data_params data = { seed, stdev };
    auto fill = [](ggml_tensor * t, void * ud) {
        set_tensor_data(t, ud);
        // Equal logits and selector scores make token 0 a deterministic accepted draft.
        if (strcmp(t->name, "output.weight") == 0 || strcmp(t->name, "selector_hidden.weight") == 0) {
            std::vector<float> zeros(ggml_nelements(t), 0.0f);
            ggml_backend_tensor_set(t, zeros.data(), 0, ggml_nbytes(t));
        }
    };
    auto mp = llama_model_default_params();
    ggml_backend_dev_t devices[] = { nullptr };
    mp.devices = devices;
    mp.n_gpu_layers = 0;
    llama_model_ptr target(llama_model_init_from_user(target_meta.get(), fill, &data, mp));
    GGML_ASSERT(target && llama_model_rope_type(target.get()) == LLAMA_ROPE_TYPE_IMROPE);

    struct capture {
        size_t zero_rows = 0;
        size_t rows = 0;
        int64_t max_rows = 0;
    } cap;
    for (bool mrope : { false, true }) {
        if (mrope) {
            const uint32_t sections[] = { 64, 0, 0, 0 };
            gguf_set_arr_data(draft_meta.get(), "dflash.rope.dimension_sections", GGUF_TYPE_UINT32, sections, 4);
        } else {
            gguf_remove_key(draft_meta.get(), "dflash.rope.dimension_sections");
        }
        llama_model_ptr draft(llama_model_init_from_user(draft_meta.get(), fill, &data, mp));
        GGML_ASSERT(draft && llama_model_dflash_selector_top_k(draft.get()) == 2);
        for (auto kv_type : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
            auto cp = llama_context_default_params();
            cp.n_ctx = 1024;
            cp.n_batch = 32;
            cp.n_ubatch = 8;
            cp.n_seq_max = 2;
            cp.n_threads = cp.n_threads_batch = 2;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
            cp.type_k = cp.type_v = kv_type;
            llama_context_ptr ctx_tgt(llama_init_from_model(target.get(), cp));
            cp.ctx_other = ctx_tgt.get();
            cp.cb_eval_user_data = &cap;
            cp.cb_eval = [](ggml_tensor * t, bool ask, void * ud) {
                if (strcmp(t->name, "inp_g_embeddings") != 0) {
                    return false;
                }
                if (ask) {
                    return true;
                }
                auto & cap = *static_cast<capture *>(ud);
                cap.rows += t->ne[1];
                std::vector<float> features(ggml_nelements(t));
                ggml_backend_tensor_get(t, features.data(), 0, ggml_nbytes(t));
                if (std::all_of(features.begin(), features.end(), [](float v) { return v == 0; })) {
                    cap.zero_rows += t->ne[1];
                }
                cap.max_rows = std::max(cap.max_rows, t->ne[1]);
                return true;
            };
            llama_context_ptr ctx_dft(llama_init_from_model(draft.get(), cp));
            GGML_ASSERT(ctx_tgt && ctx_dft);
            common_params_speculative params;
            params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH };
            params.draft.ctx_tgt = ctx_tgt.get();
            params.draft.ctx_dft = ctx_dft.get();
            params.draft.n_max = 3;
            params.draft.n_min = 1;
            common_speculative_ptr spec(common_speculative_init(params, 2));
            GGML_ASSERT(spec);
            common_speculative_set_prompt_end(nullptr, 0, 10);
            common_speculative_set_prompt_end(spec.get(), -1, 10);
            common_speculative_set_prompt_end(spec.get(), 2, 10);
            auto process = [&](const common_batch & in) {
                const int before = log.position_errors;
                const bool ok = common_speculative_process(spec.get(), in);
                GGML_ASSERT(log.position_errors == before);
                return ok;
            };
            const char * cases[] = { "text", "image+text", "image-end", "long+image", "cached+image-end", "long-text", "stale+text", "cached-tail+text", "image-span-tail", "window-boundary", "window+1" };
            const int n_tests = tail_mode < 0 ? 5 : tail_mode == 0 ? 11 : 6;
            for (int test = 0; test < n_tests; ++test) {
                llama_memory_clear(llama_get_memory(ctx_tgt.get()), true);
                llama_memory_clear(llama_get_memory(ctx_dft.get()), true);
                cap = {};
                std::vector<llama_tokens> prompts(2);
                std::vector<llama_pos> positions(2);
                size_t expected_zeros = 0;
                size_t expected_rows = 0;
                for (llama_seq_id seq = 0; seq < 2; ++seq) {
                    const int prefix = test >= 9 ? (test == 9 ? 31 : 33) + seq : (test == 5 || test == 6 ? 384 : test >= 3 ? 128 : 8) + 4*seq;
                    const bool has_image = (test >= 1 && test <= 4) || test == 8;
                    const int image_span = test == 8 ? 64 : has_image ? 3 : 0;
                    const int suffix = test == 1 || test == 3 ? 3 : test == 7 ? 16 : 0;
                    const llama_pos prompt_end = prefix + image_span + suffix;
                    const llama_pos ingest_from = window > 0 ? std::max(0, prompt_end - window) : 0;
                    if (test == 6) {
                        common_batch stale(ctx_dft.get());
                        std::vector<float> zeros(8*512, 0.0f);
                        for (int i = 0; i < 8; ++i) {
                            const llama_pos p[4] = { i, i, i, 0 };
                            stale.add_embd({ zeros.data() + i*512, 1, 512 }, p, seq, false);
                        }
                        GGML_ASSERT(llama_process(ctx_dft.get(), LLAMA_PROCESS_TYPE_DECODE, stale.get()) == 0);
                        llama_synchronize(ctx_dft.get());
                        cap.rows -= 8;
                        cap.zero_rows -= 8;
                        // Simulate a cached prefix. With full ingestion it is replayed from position 0.
                        if (!window) {
                            GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq, -1, -1));
                        }
                    }
                    if (tail_mode >= 0) {
                        common_speculative_set_prompt_end(spec.get(), seq, prompt_end);
                    }
                    expected_rows += prompt_end - ingest_from;
                    for (int pos = 0; pos < prefix;) {
                        common_batch in(ctx_tgt.get());
                        const int n = std::min(16, prefix - pos);
                        for (int i = 0; i < n; ++i) {
                            in.add(1, pos++, seq, true);
                            prompts[seq].push_back(1);
                        }
                        GGML_ASSERT(llama_process(ctx_tgt.get(), LLAMA_PROCESS_TYPE_DECODE, in.get()) == 0);
                        if (test != 4) {
                            GGML_ASSERT(process(in));
                        }
                    }
                    if (test == 4 || test == 7) {
                        for (auto * ctx : { ctx_tgt.get(), ctx_dft.get() }) {
                            if (test == 4 && ctx == ctx_dft.get()) {
                                continue;
                            }
                            const size_t size = llama_state_seq_get_size(ctx, seq);
                            std::vector<uint8_t> saved(size);
                            GGML_ASSERT(size && llama_state_seq_get_data(ctx, saved.data(), size, seq) == size);
                            GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx), seq, -1, -1));
                            GGML_ASSERT(llama_state_seq_set_data(ctx, saved.data(), size, seq) == size);
                        }
                        if (test == 4) {
                            expected_zeros += std::max(0, prefix - ingest_from);
                        }
                    }
                    llama_pos pos = prefix;
                    if (has_image) {
                        const llama_pos before = llama_memory_seq_pos_max(llama_get_memory(ctx_dft.get()), seq);
                        common_batch image(ctx_tgt.get());
                        std::vector<float> embeddings(6*256, 0.01f);
                        for (int i = 0; i < 6; ++i) {
                            const llama_pos p[4] = { pos, pos + i/3, pos + i%3, 0 };
                            image.add_embd({ embeddings.data() + i*256, 1, 256 }, p, seq, true);
                            prompts[seq].push_back(0);
                        }
                        GGML_ASSERT(llama_process(ctx_tgt.get(), LLAMA_PROCESS_TYPE_DECODE, image.get()) == 0);
                        GGML_ASSERT(process(image));
                        GGML_ASSERT(llama_memory_seq_pos_max(llama_get_memory(ctx_dft.get()), seq) == before);
                        pos += image_span;
                        expected_zeros += pos - std::max(prefix, ingest_from);
                    }
                    if (suffix) {
                        common_batch text(ctx_tgt.get());
                        for (int i = 0; i < suffix; ++i) {
                            text.add(1, pos++, seq, true);
                            prompts[seq].push_back(1);
                        }
                        GGML_ASSERT(llama_process(ctx_tgt.get(), LLAMA_PROCESS_TYPE_DECODE, text.get()) == 0);
                        GGML_ASSERT(process(text));
                    }
                    positions[seq] = pos;
                    common_speculative_begin(spec.get(), seq, prompts[seq]);
                }
                for (int round = 0; round < 2; ++round) {
                    std::vector<llama_tokens> results(2);
                    for (llama_seq_id seq = 0; seq < 2; ++seq) {
                        auto & dp = common_speculative_get_draft_params(spec.get(), seq);
                        dp.drafting = true;
                        dp.pos0 = positions[seq];
                        dp.id_last = 0;
                        dp.prompt = &prompts[seq];
                        dp.result = &results[seq];
                    }
                    const int before = log.position_errors;
                    common_speculative_draft(spec.get());
                    GGML_ASSERT(log.position_errors == before);
                    common_batch verify(ctx_tgt.get());
                    for (llama_seq_id seq = 0; seq < 2; ++seq) {
                        GGML_ASSERT(results[seq] == llama_tokens({0, 0, 0}));
                        GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq, positions[seq], -1));
                        verify.add(0, positions[seq]++, seq, true);
                        for (auto token : results[seq]) {
                            verify.add(token, positions[seq]++, seq, true);
                        }
                    }
                    GGML_ASSERT(llama_process(ctx_tgt.get(), LLAMA_PROCESS_TYPE_DECODE, verify.get()) == 0);
                    for (int i = 0; i < verify.size(); ++i) {
                        const float * logits = llama_get_logits_ith(ctx_tgt.get(), i);
                        GGML_ASSERT(std::max_element(logits, logits + 128) == logits);
                    }
                    GGML_ASSERT(process(verify));
                    for (llama_seq_id seq = 0; seq < 2; ++seq) {
                        common_speculative_accept(spec.get(), seq, 3);
                    }
                }
                llama_synchronize(ctx_dft.get());
                GGML_ASSERT(cap.zero_rows == expected_zeros && cap.max_rows <= 8);
                GGML_ASSERT(cap.rows == expected_rows + 16);
                if (test == 5 && !mrope) {
                    common_batch shifted(ctx_tgt.get());
                    for (llama_seq_id seq = 0; seq < 2; ++seq) {
                        llama_memory_seq_add(llama_get_memory(ctx_dft.get()), seq, 0, -1, 4 - positions[seq]);
                        shifted.add(1, 4, seq, false);
                    }
                    GGML_ASSERT(process(shifted));
                    llama_synchronize(ctx_dft.get());
                    GGML_ASSERT(cap.rows == expected_rows + 18);
                }
                printf("DFlash2 %s, draft-rope=%s, KV=%s, tail=%d: 12 draft tokens generated and accepted, %zu rows injected\n", cases[test], mrope ? "mrope" : "1d", ggml_type_name(kv_type), window, expected_rows);
            }
            if (window > 0) {
                llama_memory_clear(llama_get_memory(ctx_tgt.get()), true);
                llama_memory_clear(llama_get_memory(ctx_dft.get()), true);
                cap = {};
                common_speculative_set_prompt_end(spec.get(), 0, 262144);
                common_batch skipped(ctx_tgt.get());
                skipped.add(1, 0, 0, false);
                GGML_ASSERT(process(skipped)); // No target features exist: a skipped batch must not read them.
                GGML_ASSERT(llama_memory_seq_pos_max(llama_get_memory(ctx_dft.get()), 0) == -1);
                llama_tokens empty_prompt;
                llama_tokens high_result;
                common_speculative_get_draft_params(spec.get(), 1).drafting = false;
                auto & high_dp = common_speculative_get_draft_params(spec.get(), 0);
                high_dp.drafting = true;
                high_dp.pos0 = 262144;
                high_dp.id_last = 0;
                high_dp.prompt = &empty_prompt;
                high_dp.result = &high_result;
                common_speculative_draft(spec.get());
                llama_synchronize(ctx_dft.get());
                GGML_ASSERT(cap.rows == 32 && cap.zero_rows == 32 && high_result.size() == 3);
                GGML_ASSERT(llama_memory_seq_pos_min(llama_get_memory(ctx_dft.get()), 0) >= 262144 - 32);
                printf("DFlash2 262144-position media gap: only 32 rows filled\n");

                test_dflash_tail_equivalence(target_meta.get(), draft_meta.get(), mp, cp, data);
            }
            llama_set_abort_callback(ctx_dft.get(), [](void *) { return true; }, nullptr);
            common_batch next(ctx_tgt.get());
            llama_pos gap_end = llama_memory_seq_pos_max(llama_get_memory(ctx_dft.get()), 0) + 8;
            next.add(0, gap_end, 0, true);
            GGML_ASSERT(!process(next));
            llama_tokens result;
            llama_tokens prompt = { 0 };
            auto & dp = common_speculative_get_draft_params(spec.get(), 0);
            dp.drafting = true;
            dp.pos0 = llama_memory_seq_pos_max(llama_get_memory(ctx_dft.get()), 0) + 8;
            dp.id_last = 0;
            dp.prompt = &prompt;
            dp.result = &result;
            bool failed = false;
            const int before = log.position_errors;
            try {
                common_speculative_draft(spec.get());
            } catch (const std::runtime_error &) {
                failed = true;
            }
            GGML_ASSERT(failed && result.empty() && log.position_errors == before);
            llama_set_abort_callback(ctx_dft.get(), nullptr, nullptr);
            printf("DFlash2 gap fill failures propagate from process and draft\n");
        }
    }
    return 0;
}

static int test_kq_derived(size_t seed, float stdev) {
    auto meta = get_gguf_ctx(LLM_ARCH_LLAMA, false);
    auto model_and_ctx = get_model_and_ctx(meta.get(), nullptr, seed, stdev, {});
    auto * model = model_and_ctx.first.get();
    ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    GGML_ASSERT(cpu);
    int checked = 0;
    for (auto swa : {LLAMA_SWA_TYPE_NONE, LLAMA_SWA_TYPE_STANDARD, LLAMA_SWA_TYPE_CHUNKED}) {
        for (auto non_causal : {LLAMA_NON_CAUSAL_TYPE_ALL, LLAMA_NON_CAUSAL_TYPE_SWA_ONLY, LLAMA_NON_CAUSAL_TYPE_SWA_FULL}) {
            llama_hparams hp = model->hparams;
            hp.non_causal_type = non_causal;
            llama_kv_cache cache(*model, false, hp, GGML_TYPE_F16, GGML_TYPE_F16, false, false, true,
                512, 2, 8, swa, nullptr, nullptr, nullptr, nullptr);
            llama_batch_allocr alloc(1);
            const auto batch = [&](uint32_t tokens, uint32_t sequences) {
                auto ub = alloc.ubatch_reserve(tokens, sequences);
                ub.data->seq_id_data.resize(ub.n_tokens);
                for (uint32_t i = 0; i < ub.n_tokens; ++i) {
                    ub.data->seq_id_data[i] = i / tokens;
                    ub.seq_id[i] = &ub.data->seq_id_data[i];
                    ub.n_seq_id[i] = 1;
                    ub.pos[i] = i % tokens;
                }
                return ub;
            };
            auto resident = batch(32, 2);
            auto slot = cache.find_slot(resident, true);
            GGML_ASSERT(!slot.empty());
            cache.apply_ubatch(slot, resident);
            GGML_ASSERT(cache.seq_rm(0, 5, 11));
            auto query = batch(16, 1);
            for (uint32_t i = 0; i < query.n_tokens; ++i) {
                query.pos[i] = 24 + i;
            }
            GGML_ASSERT(cache.kq_mask_derivable(query));
            ggml_init_params ip = {1024 * 1024, nullptr, true};
            ggml_context_ptr ctx(ggml_init(ip));
            ggml_tensor * mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, 512, 16, 1, 1);
            ggml_tensor * cells = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 512);
            ggml_tensor * lo = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 16);
            ggml_tensor * hi = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 16);
            ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), cpu.get()));
            GGML_ASSERT(buffer);
            for (bool causal : {false, true}) {
                cache.set_input_attn_mask({mask}, &query, causal);
                cache.set_input_kq_derived(cells, lo, hi, &query, causal);
                const auto * cp = static_cast<const int32_t *>(cells->data);
                const auto * lp = static_cast<const int32_t *>(lo->data);
                const auto * hp = static_cast<const int32_t *>(hi->data);
                const auto * packed = static_cast<const ggml_fp16_t *>(mask->data);
                for (int token = 0; token < 16; ++token) {
                    for (int cell = 0; cell < 512; ++cell) {
                        const bool visible = cp[cell] != INT32_MIN && cp[cell] >= lp[token] && cp[cell] <= hp[token];
                        const float expected = ggml_fp16_to_fp32(packed[token * 512 + cell]);
                        GGML_ASSERT((visible && expected == 0.0f) || (!visible && expected == -INFINITY));
                    }
                }
                ++checked;
            }
            hp.use_alibi = true;
            GGML_ASSERT(!cache.kq_mask_derivable(query));
            hp.use_alibi = false;
            query.n_tokens = 8;
            GGML_ASSERT(!cache.kq_mask_derivable(query));
            query.n_tokens = 16;
            auto multi = batch(16, 2);
            GGML_ASSERT(!cache.kq_mask_derivable(multi));
            query.data->pos.resize(16 * 3);
            query.pos = query.data->pos.data();
            query.n_pos = 3;
            query.pos[16] = query.pos[0] + 1;
            GGML_ASSERT(!cache.kq_mask_derivable(query));
        }
    }
    printf("derived kq mask: %d packed-fill comparisons passed (SWA, non-causal, holes and foreign cells)\n", checked);
    return 0;
}

// Substitute the CPU sparse result at each QSA attention boundary, then compare full-model logits.
// The production CPU graph keeps its dense fallback; this oracle needs no dispatch override.
static int test_qsa_sparse(size_t seed, float stdev) {
    auto meta = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
    auto model_and_ctx = get_model_and_ctx(meta.get(), nullptr, seed, stdev, {});
    auto * model = model_and_ctx.first.get();
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    struct oracle {
        ggml_backend_ptr cpu;
        std::map<const ggml_tensor *, std::vector<int32_t>> indices;
        int checks = 0;
    };
    for (auto type : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        oracle ref;
        ref.cpu.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
        ggml_backend_cpu_set_n_threads(ref.cpu.get(), 2);
        ggml_backend_cpu_set_use_ref(ref.cpu.get(), true);
        auto cp = llama_context_default_params();
        cp.n_ctx = 256;
        cp.n_batch = 32;
        cp.n_ubatch = 8;
        cp.n_seq_max = 2;
        cp.n_rs_seq = 3;
        cp.kv_unified = true;
        cp.n_threads = cp.n_threads_batch = 2;
        cp.type_k = cp.type_v = type;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        llama_context_ptr dense(llama_init_from_model(model, cp));
        cp.cb_eval_user_data = &ref;
        cp.cb_eval = [](ggml_tensor * t, bool ask, void * data) {
            const bool select = t->op == GGML_OP_TOP_K_MASK;
            const ggml_tensor * mask = t->op == GGML_OP_FLASH_ATTN_EXT ? t->src[3] : nullptr;
            if (mask && mask->op == GGML_OP_RESHAPE) {
                mask = mask->src[0];
            }
            const bool attention = mask && mask->op == GGML_OP_TOP_K_MASK;
            if (ask || (!select && !attention)) { return select || attention; }
            auto & ref = *static_cast<oracle *>(data);
            ggml_context_ptr ctx(ggml_init({1024*1024, nullptr, true}));
            auto alias = [&](ggml_tensor * src) {
                auto * a = ggml_new_tensor(ctx.get(), src->type, 4, src->ne);
                memcpy(a->nb, src->nb, sizeof(a->nb));
                a->data = src->data;
                a->buffer = src->buffer;
                return a;
            };
            ggml_tensor * out;
            ggml_backend_buffer_ptr idx_buffer;
            if (select) {
                out = ggml_qsa_select(ctx.get(), alias(t->src[1]), alias(t->src[2]), alias(t->src[3]), alias(t->src[4]), t->ne[0]);
            } else {
                auto it = ref.indices.find(mask);
                GGML_ASSERT(it != ref.indices.end());
                auto * q = t->src[0];
                auto * idx = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_I32, it->second.size()/(q->ne[1]*q->ne[3]), q->ne[1], 1, q->ne[3]);
                idx_buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), ref.cpu.get()));
                ggml_backend_tensor_set(idx, it->second.data(), 0, ggml_nbytes(idx));
                float scale;
                memcpy(&scale, t->op_params, sizeof(scale));
                out = ggml_flash_attn_sparse(ctx.get(), alias(q), alias(t->src[1]), alias(t->src[2]), idx, scale);
            }
            auto * gf = ggml_new_graph_custom(ctx.get(), 32, false);
            ggml_build_forward_expand(gf, out);
            ggml_gallocr_ptr alloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(ref.cpu.get())));
            GGML_ASSERT(ggml_gallocr_alloc_graph(alloc.get(), gf));
            GGML_ASSERT(ggml_backend_graph_compute(ref.cpu.get(), gf) == GGML_STATUS_SUCCESS);
            if (select) {
                auto & idx = ref.indices[t];
                idx.resize(ggml_nelements(out));
                ggml_backend_tensor_get(out, idx.data(), 0, ggml_nbytes(out));
                const auto * mask = static_cast<const ggml_fp16_t *>(t->data);
                for (int64_t row = 0; row < ggml_nrows(t); ++row) {
                    int64_t next = 0;
                    for (int64_t cell = 0; cell < t->ne[0]; ++cell) {
                        if (ggml_fp16_to_fp32(mask[row*t->ne[0] + cell]) == 0.0f) {
                            GGML_ASSERT(next < out->ne[0] && idx[row*out->ne[0] + next++] == cell);
                        }
                    }
                    while (next < out->ne[0]) { GGML_ASSERT(idx[row*out->ne[0] + next++] == -1); }
                }
            } else {
                GGML_ASSERT(ggml_nbytes(out) == ggml_nbytes(t));
                GGML_ASSERT(memcmp(out->data, t->data, ggml_nbytes(t)) == 0);
                memcpy(t->data, out->data, ggml_nbytes(t));
                ++ref.checks;
            }
            return true;
        };
        llama_context_ptr sparse(llama_init_from_model(model, cp));
        GGML_ASSERT(dense && sparse);
        auto step = [&](int count, int pos0, int pos1) {
            for (auto * ctx : {dense.get(), sparse.get()}) {
                common_batch batch(ctx);
                for (int i = 0; i < count; ++i) {
                    for (int seq = 0; seq < 2; ++seq) {
                        const int pos = (seq == 0 ? pos0 : pos1) + i;
                        batch.add((pos*17 + seq)%n_vocab, pos, seq, true);
                    }
                }
                GGML_ASSERT(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
            }
            for (int i = 0; i < 2*count; ++i) {
                GGML_ASSERT(memcmp(llama_get_logits_ith(dense.get(), i), llama_get_logits_ith(sparse.get(), i), n_vocab*sizeof(float)) == 0);
            }
        };
        step(7, 0, 0);
        std::vector<std::vector<uint8_t>> checkpoints;
        for (auto * ctx : {dense.get(), sparse.get()}) {
            std::vector<uint8_t> state(llama_state_get_size(ctx));
            const size_t size = llama_state_get_data(ctx, state.data(), state.size());
            GGML_ASSERT(size > 0);
            state.resize(size);
            checkpoints.push_back(std::move(state));
        }
        step(4, 7, 7);
        int i = 0;
        for (auto * ctx : {dense.get(), sparse.get()}) {
            GGML_ASSERT(llama_state_set_data(ctx, checkpoints[i].data(), checkpoints[i].size()) == checkpoints[i].size());
            ++i;
        }
        for (int pos = 7; pos < 15; ++pos) { step(1, pos, pos); }
        for (auto * ctx : {dense.get(), sparse.get()}) {
            GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1));
        }
        step(3, 15, 0);

        // Use the bounded erase as the full layout rebuild reference for the open-ended trim.
        int pos[] = {18, 3};
        for (int keep : {4, 1, 2, 3, 1}) {
            step(4, pos[0], pos[1]);
            for (int seq = 0; seq < 2; ++seq) {
                const int first = pos[seq] + keep;
                GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(dense.get()), seq, first, pos[seq] + 4));
                GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(sparse.get()), seq, first, -1));
                pos[seq] = first;
            }
        }
        step(4, pos[0], pos[1]);
        GGML_ASSERT(ref.checks > 0);
        printf("Qwen4Exp QSA sparse CPU oracle: KV=%s, %d attention checks, exact logits after sequence reset, checkpoint restore and tail rollback\n",
            ggml_type_name(type), ref.checks);
    }
    return 0;
}

static int test_moe_cache_policy(size_t seed) {
#if defined(__linux__)
    namespace fs = std::filesystem;
    std::string pattern = (fs::temp_directory_path() / "llama-moe-links-XXXXXX").string();
    GGML_ASSERT(mkdtemp(&pattern[0]));
    struct cleanup {
        fs::path path;
        ~cleanup() { std::error_code ec; fs::remove_all(path, ec); }
    } fixture { pattern };
    const auto root = fixture.path / "bus/pci/devices";
    const auto fast_port = fixture.path / "devices/pci0000:00/0000:00:01.0";
    const auto slow_port = fixture.path / "devices/pci0000:00/0000:00:02.0";
    const auto bridge = slow_port / "0000:10:00.0";
    const auto fast_gpu = fast_port / "0000:03:00.0";
    const auto slow_gpu = bridge / "0000:14:00.0";
    fs::create_directories(root);
    auto link = [](const fs::path & path, const char * speed, const char * width) {
        fs::create_directories(path);
        std::ofstream(path / "max_link_speed") << speed;
        std::ofstream(path / "current_link_width") << width;
    };
    link(fast_port, "16.0 GT/s PCIe\n", "16\n");
    link(slow_port, "16.0 GT/s PCIe\n", "4\n");
    link(bridge, "16.0 GT/s PCIe\n", "16\n");
    link(fast_gpu, "16.0 GT/s PCIe\n", "16\n");
    link(slow_gpu, "16.0 GT/s PCIe\n", "16\n");
    fs::create_directory_symlink(fast_gpu, root / "0000:03:00.0");
    fs::create_directory_symlink(slow_gpu, root / "0000:14:00.0");
    const auto pci_root = root.string();
    auto probe = [&](const char * id) { return llama_moe_cache_host_link(id, pci_root.c_str()); };
    auto fast = probe("0000:03:00.0");
    auto slow = probe("0000:14:00.0");
    GGML_ASSERT(fast.bandwidth() == 256 && fast.limiting_device == "0000:03:00.0");
    GGML_ASSERT(probe("0:3:0.0").bandwidth() == 256);
    GGML_ASSERT(slow.bandwidth() == 64 && slow.limiting_device == "0000:00:02.0");
    GGML_ASSERT(moe_cache_stage_on_faster_link(slow.bandwidth(), fast.bandwidth(), 260, 512));
    std::ofstream(fast_gpu / "current_link_speed") << "2.5 GT/s PCIe\n";
    GGML_ASSERT(probe("0000:03:00.0").bandwidth() == 256);
    // Sysfs numbers retain the C locale even in an application with comma decimals.
    struct comma_decimal : std::numpunct<char> {
        char do_decimal_point() const override { return ','; }
    };
    const auto previous_locale = std::locale();
    std::locale::global(std::locale(previous_locale, new comma_decimal));
    GGML_ASSERT(probe("0000:03:00.0").bandwidth() == 256);
    std::locale::global(previous_locale);
    for (const char * id : {static_cast<const char *>(nullptr), "", "0000:03:00.0-v0", "0000:03:20.0", "0000:03:00.8", "../0000:03:00.0", "0000:04:00.0"}) {
        GGML_ASSERT(probe(id).bandwidth() == 0);
    }
    for (const char * width : {"", "0", "-1", "33", "unknown"}) {
        link(slow_port, "16.0 GT/s PCIe\n", width);
        GGML_ASSERT(probe("0000:14:00.0").bandwidth() == 0);
    }
    for (const char * speed : {"", "Unknown speed", "nan GT/s PCIe", "256.0 GT/s PCIe", "16.0 GB/s"}) {
        link(slow_port, speed, "4\n");
        GGML_ASSERT(probe("0000:14:00.0").bandwidth() == 0);
    }
    link(slow_port, "16.0 GT/s PCIe\n", "4\n");
    fs::remove(bridge / "current_link_width");
    GGML_ASSERT(probe("0000:14:00.0").bandwidth() == 0);
    link(bridge, "8.0 GT/s PCIe\n", "4\n");
    GGML_ASSERT(probe("0000:14:00.0").bandwidth() == 32);
    link(slow_port, "16.0 GT/s PCIe\n", "16\n");
    link(bridge, "16.0 GT/s PCIe\n", "16\n");
    GGML_ASSERT(!moe_cache_stage_on_faster_link(probe("0000:14:00.0").bandwidth(), fast.bandwidth(), 260, 512));
    printf("MoE PCIe topology: endpoint, upstream bottleneck, locale and unknown-link fallback passed\n");
#endif
    GGML_ASSERT(moe_cache_stage_on_faster_link(64, 256, 260, 512));
    GGML_ASSERT(!moe_cache_stage_on_faster_link(256, 256, 260, 512));
    GGML_ASSERT(!moe_cache_stage_on_faster_link(64, 256, 400, 512));
    GGML_ASSERT(!moe_cache_stage_on_faster_link(64, 256, 512, 512));
    GGML_ASSERT(!moe_cache_stage_on_faster_link(0, 256, 260, 512));
    GGML_ASSERT(!moe_cache_stage_on_faster_link(64, 0, 260, 512));
    GGML_ASSERT(!moe_cache_fits(0, 2, 512, 1024));
    GGML_ASSERT(!moe_cache_fits(1024, 0, 512, 1024));
    GGML_ASSERT(!moe_cache_fits(1, 513, 512, 1024));
    GGML_ASSERT(moe_cache_fits(32, 2, 512, 64));
    GGML_ASSERT(!moe_cache_fits(33, 2, 512, 64));
    GGML_ASSERT(moe_cache_fits(INT64_MAX, 10, 512, 512));
    GGML_ASSERT(!moe_cache_fits(INT64_MAX, 10, 512, 511));

    moe_cache_clock clock;
    clock.begin(0, 0, 7); GGML_ASSERT(clock.tokens == 8);
    clock.begin(0, 4, 7); GGML_ASSERT(clock.tokens == 8); // repeated verify / rollback
    clock.begin(0, 7, 8); GGML_ASSERT(clock.tokens == 9);
    clock.begin(1, 20, 23); GGML_ASSERT(clock.tokens == 13); // distinct sequence, cached prefix
    clock.begin(0, 0, 3); GGML_ASSERT(clock.tokens == 17); // new prompt
    GGML_ASSERT(clock.latency_tokens == 17);
    clock.begin(0, 4, 1027, false);
    GGML_ASSERT(clock.tokens == 1041 && clock.latency_tokens == 17);
    clock.begin(0, 1024, 1027);
    GGML_ASSERT(clock.latency_tokens == 17); // speculative revisit
    clock.begin(0, 1026, 1031);
    GGML_ASSERT(clock.latency_tokens == 21);

    moe_cache_policy cache;
    constexpr int layers = 3, experts = 16, slots = 10;
    std::vector<std::vector<int32_t>> maps(layers, std::vector<int32_t>(experts, -1));
    cache.init(layers, experts, slots);
    for (int il = 0; il < layers; ++il) cache.slot_map[il] = maps[il].data();
    std::vector<moe_cache_policy::fill> fills;
    size_t hits = 0;
    std::mt19937 gen(seed);
    moe_cache_policy scoring; scoring.init(1, 1, 1);
    for (int i = 0; i < 4096; ++i) {
        const uint64_t history = (uint64_t(gen()) << 32) | gen();
        const int delta = gen() % 32;
        scoring.needs[0].history = history; scoring.needs[0].slice = 0;
        const uint64_t aged = delta >= 16 ? 0 : history << (4*delta);
        uint32_t expected = 0;
        for (int n = 0; n < 16; ++n) expected += ((aged >> (4*n)) & 15)*(n < 4 ? 4 : 1);
        GGML_ASSERT(scoring.score(0, delta) == expected);
    }
    for (int step = 0; step < 2000; ++step) {
        const int il = gen() % layers;
        const bool force = step % 3 == 0;
        std::vector<int32_t> ids(1 + gen() % (force ? 8 : 128));
        for (auto & id : ids) id = gen() % experts;
        const auto before_maps = maps;
        const auto before_keys = cache.key_of;
        const auto before = maps[il];
        std::vector<bool> selected(experts, false), filled(experts, false), temporary(experts, false);
        for (auto id : ids) selected[id] = true;
        size_t expected_hits = 0;
        for (int e = 0; e < experts; ++e) expected_hits += selected[e] && before[e] >= 0;
        GGML_ASSERT(cache.plan(il, ids.data(), ids.size(), fills, hits, step*37, force));
        GGML_ASSERT(hits == expected_hits);
        // Full-sort oracle, independent of the planner's heap and stored scores.
        const auto expected_score = [&](int32_t key) {
            const auto & d = cache.needs[key];
            const auto age = [](uint64_t history, uint64_t previous, uint64_t now) {
                const uint64_t delta = now - previous;
                history = delta >= 16 ? 0 : history << (4*delta);
                uint32_t result = 0;
                for (int n = 0; n < 16; ++n) result += ((history >> (4*n)) & 15)*(n < 4 ? 4 : 1);
                return result;
            };
            return age(d.latency_history, d.latency_slice, 0)*(moe_cache_policy::window_max_score + 1) +
                age(d.history, d.slice, step*37/moe_cache_policy::slice_tokens);
        };
        std::vector<int32_t> victims, misses, destinations;
        for (int s = 0; s < slots; ++s) {
            const int key = before_keys[s];
            if (key < 0 || key/experts != il || !selected[key%experts]) victims.push_back(s);
        }
        std::sort(victims.begin(), victims.end(), [&](int a, int b) {
            const int ka = before_keys[a], kb = before_keys[b];
            if ((ka < 0) != (kb < 0)) return ka < 0;
            if (ka < 0) return a < b;
            if (expected_score(ka) != expected_score(kb)) return expected_score(ka) < expected_score(kb);
            if (cache.needs[ka].stamp != cache.needs[kb].stamp) return cache.needs[ka].stamp < cache.needs[kb].stamp;
            if (cache.needs[ka].frequency != cache.needs[kb].frequency) return cache.needs[ka].frequency < cache.needs[kb].frequency;
            return a < b;
        });
        for (int e = 0; e < experts; ++e) if (selected[e] && before[e] < 0) misses.push_back(e);
        std::sort(misses.begin(), misses.end(), [&](int a, int b) {
            const int ka = il*experts + a, kb = il*experts + b;
            if (expected_score(ka) != expected_score(kb)) return expected_score(ka) > expected_score(kb);
            if (cache.needs[ka].frequency != cache.needs[kb].frequency) return cache.needs[ka].frequency > cache.needs[kb].frequency;
            return a < b;
        });
        std::vector<int32_t> admitted;
        for (int e : misses) {
            if (destinations.size() == victims.size()) continue;
            const int s = victims[destinations.size()], key = before_keys[s];
            if (!force && key >= 0 && expected_score(il*experts + e) <= expected_score(key) + std::max(1u, expected_score(key)/8)) continue;
            admitted.push_back(e);
            destinations.push_back(s);
        }
        std::sort(admitted.begin(), admitted.end());
        std::sort(destinations.begin(), destinations.end());
        GGML_ASSERT(fills.size() == admitted.size());
        for (size_t i = 0; i < fills.size(); ++i) {
            GGML_ASSERT(fills[i].expert == admitted[i] && fills[i].slot == destinations[i]);
        }
        auto patched = before_maps;
        std::vector<bool> changed(layers*experts, false);
        for (int32_t key : cache.changed) {
            GGML_ASSERT(key >= 0 && key < layers*experts && !changed[key]);
            changed[key] = true;
            patched[key/experts][key%experts] = maps[key/experts][key%experts];
        }
        GGML_ASSERT(patched == maps && cache.changed.size() <= 2*fills.size());
        for (const auto & fill : fills) {
            GGML_ASSERT(selected[fill.expert] && before[fill.expert] < 0 && !filled[fill.expert]);
            GGML_ASSERT(maps[il][fill.expert] == fill.slot);
            filled[fill.expert] = true;
        }
        for (auto e : cache.temporary) { GGML_ASSERT(!temporary[e]); temporary[e] = true; }
        for (int l = 0; l < layers; ++l) for (int e = 0; e < experts; ++e) {
            const int slot = maps[l][e];
            if (slot >= 0) GGML_ASSERT(slot < slots && cache.key_of[slot] == l*experts + e);
            if (l == il && selected[e]) {
                GGML_ASSERT((slot >= 0) != temporary[e]);
                GGML_ASSERT(before[e] < 0 ? (filled[e] || temporary[e]) : slot == before[e]);
            }
        }
        for (int s = 0; s < slots; ++s) {
            const int key = cache.key_of[s];
            GGML_ASSERT(key < 0 || maps[key/experts][key % experts] == s);
        }
    }
    // One hundred assignments in one batch still count as one weight request.
    moe_cache_policy tiny; tiny.init(1, experts, 2);
    std::vector<int32_t> tiny_map(experts, -1); tiny.slot_map[0] = tiny_map.data();
    int32_t initial[] = {0, 1};
    GGML_ASSERT(tiny.plan(0, initial, 2, fills, hits, 1, false));
    const uint64_t history = tiny.needs[0].history;
    for (int retry = 0; retry < 100; ++retry) tiny.plan(0, initial, 2, fills, hits, 1, false);
    GGML_ASSERT(tiny.needs[0].history == history);
    std::vector<int32_t> scan(100, 2);
    tiny.plan(0, scan.data(), scan.size(), fills, hits, 2, false);
    GGML_ASSERT(fills.empty() && tiny.temporary == std::vector<int32_t>{2});
    tiny.plan(0, scan.data(), scan.size(), fills, hits, 3, false);
    GGML_ASSERT(fills.size() == 1 && tiny_map[2] >= 0); // repeated need wins admission
    int32_t excessive[] = {0, 1, 2};
    const auto prior = tiny_map;
    GGML_ASSERT(!tiny.plan(0, excessive, 3, fills, hits, 4, true) && prior == tiny_map);
    tiny.plan(0, excessive, 3, fills, hits, 5, false); // over-capacity selection streams safely
    GGML_ASSERT(tiny.temporary.size() == 1);
    tiny.plan(0, initial, 2, fills, hits, 20000, false); // expiry
    GGML_ASSERT(tiny.needs[2].history == 0);
    // A full prompt must not expire decode demand or upload into a remote bank.
    moe_cache_policy phased; phased.init(1, experts, 2);
    std::vector<int32_t> phased_map(experts, -1); phased.slot_map[0] = phased_map.data();
    phased.plan(0, initial, 2, fills, hits, 1, false, 0);
    phased.plan(0, initial, 2, fills, hits, 1, true, 0);
    GGML_ASSERT(phased.needs[0].latency_history == 1); // separate prefill stamp
    for (int retry = 0; retry < 4; ++retry) phased.plan(0, initial, 2, fills, hits, 1, true, 0);
    GGML_ASSERT(phased.needs[0].latency_history == 1);
    for (int batch = 1; batch <= 10; ++batch) {
        phased.plan(0, scan.data(), scan.size(), fills, hits, 1 + batch*1024, false, 0);
        GGML_ASSERT(fills.empty() && phased_map[0] >= 0 && phased_map[1] >= 0);
    }
    const auto resident = phased_map;
    phased.plan(0, excessive, 3, fills, hits, 11000, false, 0, false);
    GGML_ASSERT(hits == 0 && fills.empty() && phased_map == resident);
    GGML_ASSERT(phased.temporary == std::vector<int32_t>({0, 1, 2}));
    GGML_ASSERT(phased.score(0, 11000/moe_cache_policy::slice_tokens, 0) > 0);
    GGML_ASSERT(phased.score(0, 11000/moe_cache_policy::slice_tokens, 16) <= moe_cache_policy::window_max_score);
    GGML_ASSERT(phased.needs[0].latency_history == 0);
    // A delayed device hit must observe demand without reviving an evicted slot.
    moe_cache_policy delayed; delayed.init(1, experts, 2);
    std::vector<int32_t> delayed_map(experts, -1); delayed.slot_map[0] = delayed_map.data();
    delayed.plan(0, initial, 2, fills, hits, 1, true, 1);
    int32_t replacement[] = {2, 3};
    delayed.plan(0, replacement, 2, fills, hits, 2, true, 2);
    const auto after_eviction = delayed_map;
    GGML_ASSERT(delayed_map[0] < 0 && delayed_map[1] < 0);
    delayed.plan(0, initial, 2, fills, hits, 2, true, 2, false);
    GGML_ASSERT(delayed_map == after_eviction && fills.empty());
    GGML_ASSERT(delayed.changed.empty());
    GGML_ASSERT(delayed.needs[0].latency_history > 0);
    printf("MoE window policy: phase clocks, latency protection, remote staging, link selection, exact victim oracle and sparse map updates for 2000 mixed plans passed\n");
    return 0;
}

static int test_moe_cache_prefill(size_t seed, float stdev, const char * target_backend) {
    std::vector<ggml_backend_dev_t> devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        if (target_backend && strcmp(target_backend, ggml_backend_dev_name(dev)) != 0) { continue; }
        auto reg = ggml_backend_dev_backend_reg(dev);
        using supports_fn = bool (*)(ggml_backend_dev_t);
        auto supports = reg ? (supports_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hip_moe_cache_supports_prefill") : nullptr;
        if (supports && supports(dev)) { devices.push_back(dev); }
    }
    const bool cpu_fixture = devices.empty();
    if (cpu_fixture) {
        devices.push_back(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
        GGML_ASSERT(devices.back());
        printf("No supported RDNA3 HIP device: check the CPU fixture; GPU cache execution is SKIPPED\n");
    }
    devices.resize(std::min<size_t>(devices.size(), 2));
    auto meta = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
    gguf_set_val_u32(meta.get(), "qwen4exp.expert_count", 16);
    gguf_set_val_u32(meta.get(), "qwen4exp.expert_feed_forward_length", 640);
    ggml_context_ptr types(ggml_init({16*ggml_tensor_overhead(), nullptr, true}));
    for (int il = 0; il < 2; ++il) {
        for (const char * bank : {"gate_up", "down"}) {
            const bool down = strcmp(bank, "down") == 0;
            auto * t = ggml_new_tensor_3d(types.get(), GGML_TYPE_Q8_0, down ? 640 : 256, down ? 256 : 1280, 16);
            ggml_format_name(t, "blk.%d.ffn_%s_exps.weight", il, bank);
            gguf_add_tensor(meta.get(), t);
        }
    }
    const llama_model_tensor_buft_override overrides[] = {
        { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() }, { nullptr, nullptr },
    };
    auto mp = llama_model_default_params();
    auto terminated = devices;
    terminated.push_back(nullptr);
    mp.devices = terminated.data();
    mp.n_gpu_layers = cpu_fixture ? 0 : 999;
    mp.tensor_buft_overrides = overrides;
    mp.progress_callback = silent_model_load_progress;
    tensor_data_params data = {seed, stdev};
    auto fill = [](ggml_tensor * t, void * userdata) {
        if (t->type != GGML_TYPE_Q8_0) { set_tensor_data(t, userdata); return; }
        std::vector<float> values(ggml_nelements(t));
        fill_normal(values, t->name, *static_cast<tensor_data_params *>(userdata));
        std::vector<uint8_t> quantized(ggml_nbytes(t));
        GGML_ASSERT(ggml_quantize_chunk(t->type, values.data(), quantized.data(), 0, ggml_nrows(t), t->ne[0], nullptr) == quantized.size());
        ggml_backend_tensor_set(t, quantized.data(), 0, quantized.size());
    };
    llama_model_ptr model(llama_model_init_from_user(meta.get(), fill, &data, mp));
    GGML_ASSERT(model);
    struct capture {
        llama_context * ctx = nullptr;
        const llama_model * model = nullptr;
        bool prefill = false;
        bool check_locality = true;
        int cached = 0;
        int redirected = 0;
        int fallback = 0;
        int device_lookups = 0;
        int device_requests = 0;
        int last_layer_device_lookups = 0;
        int cropped_lookups = 0;
    };
    // Compare the same GPU model with and without caching, with equal and small/zero budgets.
    for (int config = 0; config < (cpu_fixture ? 1 : 3); ++config) {
        std::vector<size_t> budgets(llama_max_devices(), 0);
        for (size_t i = 0; i < devices.size(); ++i) { budgets[i] = cpu_fixture ? 0 : 8*1024*1024; }
        if (config == 1) { budgets[devices.size() - 1] = 4*1024*1024; }
        if (config == 2) { budgets[0] = 0; }
        auto cp = llama_context_default_params();
        cp.n_ctx = 2048;
        cp.n_batch = cp.n_ubatch = 1024;
        cp.n_threads = cp.n_threads_batch = 2;
        cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        // Observe both contexts at the same operation boundaries.
        cp.cb_eval = [](ggml_tensor * t, bool ask, void * userdata) {
            if (ask) { return (t->op == GGML_OP_MUL_MAT_ID || t->op == GGML_OP_MUL_MAT_ID_STREAM || t->op == GGML_OP_MOE_CACHE_LOOKUP); }
            auto & cap = *static_cast<capture *>(userdata);
            if (t->op == GGML_OP_MOE_CACHE_LOOKUP) {
                if (ggml_moe_get_lookup_stage(t) == GGML_MOE_LOOKUP_PUBLISH) {
                    ++cap.device_requests;
                    return true;
                }
                ++cap.device_lookups;
                int il = -1;
                GGML_ASSERT(sscanf(t->name, "ffn_moe_slots-%d", &il) == 1);
                if (il == int(cap.model->layers.size()) - 1) ++cap.last_layer_device_lookups;
                if (cap.prefill) ++cap.cropped_lookups;
                return true;
            }
            if (cap.check_locality && cap.prefill && t->ne[2] > moe_cache_max_batch) {
                const char * name = t->src[0]->name;
                const char * block = strstr(name, "blk.");
                int il = -1;
                GGML_ASSERT(block && sscanf(block, "blk.%d.", &il) == 1);
                if (strncmp(name, "moe_cache.", 10) == 0) {
                    auto backend = ggml_backend_sched_get_tensor_backend(cap.ctx->get_sched(), t);
                    const auto * w = t->src[0];
                    const auto buffer = w->view_src ? w->view_src->buffer : w->buffer;
                    const auto execution = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer));
                    GGML_ASSERT(backend && ggml_backend_get_device(backend) == execution);
                    if (strncmp(name, "moe_cache.stream.", 17) == 0) {
                        GGML_ASSERT(t->op == GGML_OP_MUL_MAT_ID_STREAM && execution != cap.model->dev_layer(il));
                        GGML_ASSERT(w->view_src && w->ne[2] == moe_cache_stage_slots);
                        GGML_ASSERT(ggml_nbytes(w) <= ggml_nbytes(w->view_src));
                        ++cap.redirected;
                    } else {
                        GGML_ASSERT(execution == cap.model->dev_layer(il));
                    }
                    ++cap.cached;
                } else {
                    ++cap.fallback;
                }
            }
            return true;
        };
        capture cap;
        cap.model = model.get();
        cap.check_locality = !cpu_fixture;
        // Reference placement follows the stock scheduler and is not constrained by cache ownership.
        auto reference_cp = cp;
        reference_cp.cb_eval = [](ggml_tensor * t, bool ask, void *) {
            return ask ? (t->op == GGML_OP_MUL_MAT_ID || t->op == GGML_OP_MUL_MAT_ID_STREAM) : true;
        };
        llama_context_ptr reference(llama_init_from_model(model.get(), reference_cp));
        cp.cb_eval_user_data = &cap;
        cp.moe_cache_split = budgets.data();
        llama_context_ptr actual(llama_init_from_model(model.get(), cp));
        GGML_ASSERT(actual && reference);
        cap.ctx = actual.get();
        const auto actual_memory = actual->memory_breakdown();
        const auto reference_memory = reference->memory_breakdown();
        for (size_t i = 0; i < devices.size(); ++i) {
            auto buft = ggml_backend_dev_buffer_type(devices[i]);
            const size_t allocated = actual_memory.at(buft).context - reference_memory.at(buft).context;
            // Resident banks stay within budget. Two fixed packet lanes and
            // mirrored slot maps are additional, separately bounded memory.
            size_t stride = 0;
            for (int il = 0; il < int(model->layers.size()); ++il) {
                for (const auto * w : {model->layers[il].ffn_gate_up_exps, model->layers[il].ffn_down_exps}) {
                    if (w) stride = std::max(stride, w->nb[2]);
                }
            }
            const size_t alignment = ggml_backend_buft_get_alignment(buft);
            const size_t scratch = 2*(moe_cache_stage_slots + 1)*stride + 2*16*(sizeof(uint64_t) + sizeof(int32_t)) +
                model->layers.size()*16*sizeof(int32_t) + 8*alignment;
            GGML_ASSERT(allocated <= budgets[i] + scratch);
        }
        llama_pos pos = 0, ref_pos = 0;
        double worst_nmse = 0;
        auto step = [&](int tokens, int outputs = -1) {
            if (outputs < 0) outputs = tokens;
            cap.prefill = tokens > moe_cache_max_batch;
            const auto expected = recovery_decode(reference.get(), ref_pos, tokens, outputs, false, false);
            const auto result = recovery_decode(actual.get(), pos, tokens, outputs, false, false);
            const auto error = nmse(expected, result);
            GGML_ASSERT(std::isfinite(error) && error < 5e-4);
            worst_nmse = std::max(worst_nmse, error);
        };
        // A cold large prefill with one output still has a small last-layer lookup.
        step(64, 1);
        std::vector<std::vector<uint8_t>> checkpoint;
        for (auto * ctx : {reference.get(), actual.get()}) {
            std::vector<uint8_t> state(llama_state_get_size(ctx));
            const size_t size = llama_state_get_data(ctx, state.data(), state.size());
            GGML_ASSERT(size > 0);
            state.resize(size);
            checkpoint.push_back(std::move(state));
        }
        for (int width : {1, 4, 5, 7, 8}) { step(width); }
        int index = 0;
        for (auto * ctx : {reference.get(), actual.get()}) {
            const auto & state = checkpoint[index++];
            GGML_ASSERT(llama_state_set_data(ctx, state.data(), state.size()) == state.size());
        }
        pos = ref_pos = 64;
        step(8);
        for (int width : {512, 1024}) {
            for (auto * ctx : {reference.get(), actual.get()}) { llama_memory_clear(llama_get_memory(ctx), true); }
            pos = ref_pos = 0;
            step(width);
            step(4);
        }
        const bool any_budget = std::any_of(budgets.begin(), budgets.end(), [](size_t b) { return b != 0; });
        GGML_ASSERT(!any_budget || config != 0 || cap.cached > 0);
        GGML_ASSERT(cap.device_requests <= cap.device_lookups);
        GGML_ASSERT(cap.last_layer_device_lookups == 0 || cap.cropped_lookups > 0);
        printf("MoE prefill %s model: devices=%zu, budgets=%d, cached=%d, redirected=%d, fallback=%d, device lookups=%d, split requests=%d, cropped=%d, worst logit NMSE=%g, checkpoint and warm-cache reset passed\n",
            cpu_fixture ? "CPU fixture" : "GPU", devices.size(), config, cap.cached, cap.redirected, cap.fallback, cap.device_lookups, cap.device_requests, cap.cropped_lookups, worst_nmse);
    }
    return 0;
}

static int test_layer_inputs(size_t seed, float stdev) {
    auto meta = get_gguf_ctx(LLM_ARCH_QWEN4EXP, true);
    auto model_and_ctx = get_model_and_ctx(meta.get(), nullptr, seed, stdev, {});
    auto * model = model_and_ctx.first.get();
    const int n_layer = llama_model_n_layer(model);
    const int n_embd = llama_model_n_embd(model);
    const auto tokens = get_tokens(11, 128, seed);

    struct capture {
        int n_layer;
        std::vector<std::vector<float>> means;
    };

    for (auto kv_type : { GGML_TYPE_F16, GGML_TYPE_Q8_0 }) {
        capture cap = { n_layer, std::vector<std::vector<float>>(n_layer) };
        auto cp = llama_context_default_params();
        cp.n_ctx = 128;
        cp.n_batch = 32;
        cp.n_ubatch = 4;
        cp.n_threads = cp.n_threads_batch = 2;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = cp.type_v = kv_type;
        llama_context_ptr reference(llama_init_from_model(model, cp));
        GGML_ASSERT(reference);
        cp.cb_eval_user_data = &cap;
        cp.cb_eval = [](ggml_tensor * t, bool ask, void * data) {
            auto & cap = *static_cast<capture *>(data);
            int il = -1;
            if (strcmp(t->name, "hc_init") == 0) {
                il = 0;
            } else {
                int layer = -1;
                if (sscanf(t->name, "l_last-%d", &layer) == 1 && t->name == "l_last-" + std::to_string(layer)) {
                    il = layer + 1;
                }
            }
            if (il < 0 || il >= cap.n_layer) {
                return false;
            }
            if (ask) {
                return true;
            }
            GGML_ASSERT(t->type == GGML_TYPE_F32 && t->ne[1] > 1);
            std::vector<float> streams(ggml_nbytes(t) / sizeof(float));
            ggml_backend_tensor_get(t, streams.data(), 0, ggml_nbytes(t));
            for (int64_t token = 0; token < t->ne[2]; ++token) {
                for (int64_t col = 0; col < t->ne[0]; ++col) {
                    float sum = 0;
                    for (int64_t s = 0; s < t->ne[1]; ++s) {
                        sum += streams[(token*t->nb[2] + s*t->nb[1] + col*t->nb[0]) / sizeof(float)];
                    }
                    cap.means[il].push_back(sum / t->ne[1]);
                }
            }
            return true;
        };
        llama_context_ptr ctx(llama_init_from_model(model, cp));
        GGML_ASSERT(ctx);
        for (int il = 0; il < n_layer; ++il) {
            llama_set_embeddings_layer_inp(ctx.get(), il, true);
        }
        llama_set_embeddings_nextn(ctx.get(), true, false);
        llama_set_embeddings_nextn(reference.get(), true, false);
        const auto expected_logits = get_logits(model, reference.get(), tokens);
        const auto actual_logits = get_logits(model, ctx.get(), tokens);
        GGML_ASSERT(nmse(expected_logits, actual_logits) < 1e-10);
        for (int il = 0; il < n_layer; ++il) {
            GGML_ASSERT(cap.means[il].size() == tokens.size()*n_embd);
            const float * actual = llama_get_embeddings_layer_inp(ctx.get(), il);
            for (size_t i = 0; i < cap.means[il].size(); ++i) {
                GGML_ASSERT(std::isfinite(actual[i]) && fabsf(actual[i] - cap.means[il][i]) < 1e-5f);
            }
        }
        const size_t n_nextn = tokens.size()*llama_model_n_embd_out(model);
        const auto * actual_nextn = llama_get_embeddings_nextn(ctx.get());
        const auto * expected_nextn = llama_get_embeddings_nextn(reference.get());
        GGML_ASSERT(memcmp(actual_nextn, expected_nextn, n_nextn*sizeof(float)) == 0);
        printf("qwen4exp layer inputs: stream means, chunked extraction and MTP state KV=%s passed\n", ggml_type_name(kv_type));
    }

    std::vector<size_t> sizes(llama_max_devices(), 0);
    auto cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_threads = cp.n_threads_batch = 2;
    cp.moe_cache_size = 1024 * 1024;
    cp.moe_cache_split = sizes.data();
    llama_context_ptr disabled(llama_init_from_model(model, cp));
    GGML_ASSERT(disabled);
    sizes[0] = std::numeric_limits<size_t>::max();
    sizes[1] = 1;
    llama_context_ptr overflow(llama_init_from_model(model, cp));
    GGML_ASSERT(!overflow);
    printf("MoE cache parameters: zero list overrides total, overflowing list rejected\n");
    return 0;
}

static int test_graph_recovery(size_t seed, float stdev) {
    recovery_log log;
    test_text_plan_retention(seed, stdev, log);
    test_memory_update_reservation(seed, stdev, log);
    for (int config = 0; config < 6; ++config) {
        const bool mtp = config % 3 == 2;
        const bool quantized = config >= 3;
        const llm_arch arch = config % 3 == 0 ? LLM_ARCH_LLAMA : LLM_ARCH_QWEN35;
        auto meta = get_gguf_ctx(arch, false);
        if (arch == LLM_ARCH_QWEN35) {
            gguf_set_val_u32(meta.get(), "qwen35.full_attention_interval", 2);
        }
        if (mtp) {
            gguf_set_val_u32(meta.get(), "qwen35.nextn_predict_layers", 1);
            gguf_set_val_u32(meta.get(), "qwen35.block_count", 3);
        }
        auto mp = llama_model_default_params();
        ggml_backend_dev_t devices[] = { nullptr };
        mp.devices = devices;
        mp.n_gpu_layers = 0;
        mp.load_mtp = mtp;
        tensor_data_params data = { seed, stdev };
        llama_model_ptr model(llama_model_init_from_user(meta.get(), set_tensor_data, &data, mp));
        GGML_ASSERT(model);
        auto cp = llama_context_default_params();
        cp.n_ctx = 512;
        cp.n_batch = cp.n_ubatch = 64;
        cp.n_outputs_max = cp.n_outputs_max_per_seq = mtp ? 1 : 49;
        cp.n_threads = cp.n_threads_batch = 2;
        cp.ctx_type = mtp ? LLAMA_CONTEXT_TYPE_MTP : LLAMA_CONTEXT_TYPE_DEFAULT;
        if (quantized) {
            cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
        llama_context_ptr reference(llama_init_from_model(model.get(), cp));
        GGML_ASSERT(ctx && reference);
        if (mtp) {
            llama_set_embeddings_nextn(ctx.get(), true, true);
            llama_set_embeddings_nextn(reference.get(), true, true);
        }
        llama_pos pos = 0;
        recovery_decode(ctx.get(), pos, 4, 1, false, mtp);
        llama_pos ref_start = 0;
        recovery_decode(reference.get(), ref_start, 4, 1, false, mtp);
        ggml_backend_sched_t scheduler = ctx->get_sched();
        const size_t buffer_size = ggml_backend_sched_get_buffer_size(scheduler, ggml_backend_sched_get_backend(scheduler, 0));
        for (int transition = 0; transition < 2; ++transition) {
            recovery_decode(ctx.get(), pos, 3, 0, true, mtp, transition == 1 && !mtp);
            // Restore the same sequence into a context that still has its initial text plan.
            const size_t size = llama_state_seq_get_size(ctx.get(), 0);
            std::vector<uint8_t> checkpoint(size);
            GGML_ASSERT(size && llama_state_seq_get_data(ctx.get(), checkpoint.data(), size, 0) == size);
            GGML_ASSERT(llama_state_seq_set_data(reference.get(), checkpoint.data(), size, 0) == size);
            llama_pos ref_pos = pos;
            for (int step = 0; step < 9; ++step) {
                const int tokens = step == 0 ? 33 : step % 2 ? 1 : 4;
                const int outputs = step % 2 ? 1 : 0;
                const int before = log.reserves;
                auto actual = recovery_decode(ctx.get(), pos, tokens, outputs, false, mtp);
                GGML_ASSERT(log.reserves - before == (step == 0 ? 1 : 0));
                const int before_ref = log.reserves;
                auto expected = recovery_decode(reference.get(), ref_pos, tokens, outputs, false, mtp);
                GGML_ASSERT(log.reserves == before_ref);
                GGML_ASSERT(actual == expected);
            }
            if (!mtp) {
                const int before = log.reserves;
                auto actual = recovery_decode(ctx.get(), pos, 49, 49, false, mtp);
                auto expected = recovery_decode(reference.get(), ref_pos, 49, 49, false, mtp);
                GGML_ASSERT(actual == expected && log.reserves == before);
            }
            const auto reused = llama_perf_context(ctx.get()).n_reused;
            const int before = log.reserves;
            for (int i = 0; i < 4; ++i) {
                auto actual = recovery_decode(ctx.get(), pos, 1, 1, false, mtp);
                auto expected = recovery_decode(reference.get(), ref_pos, 1, 1, false, mtp);
                GGML_ASSERT(actual == expected);
            }
            GGML_ASSERT(log.reserves == before && llama_perf_context(ctx.get()).n_reused > reused);
            GGML_ASSERT(ctx->get_sched() == scheduler);
            GGML_ASSERT(ggml_backend_sched_get_buffer_size(scheduler, ggml_backend_sched_get_backend(scheduler, 0)) == buffer_size);
            GGML_ASSERT(llama_perf_context(ctx.get()).n_reallocs_same == 0);
        }
        printf("graph recovery: %s%s KV=%s passed\n", llama_arch_name(arch), mtp ? " MTP" : "", quantized ? "q8_0" : "f16");
    }
    return 0;
}

#endif

struct kv_shift_move {
    llama_pos p0;
    llama_pos p1;
    llama_pos delta;
};

static void decode_range(llama_context * lctx, const llama_token * tokens, uint32_t n_tokens, llama_pos pos0, llama_seq_id seq_id, bool output) {
    common_batch batch(lctx);
    for (uint32_t i = 0; i < n_tokens; i++) {
        batch.add(tokens[i], pos0 + i, seq_id, output || i == n_tokens - 1);
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode batch");
    }
}

// the raw attention scores of each layer are linear in K, unlike the logits they do not depend on how sharp the attention is
struct kq_capture {
    bool active = false;
    std::vector<std::pair<std::string, std::vector<float>>> data;
};

// the scores are compared cell by cell, so the cells of the probe must match: prefix and probe fit in the SWA window,
// otherwise the probe can reuse a masked SWA cell in one run and not in the other
static bool kq_capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    kq_capture * cap = (kq_capture *) user_data;
    const char * name = ggml_get_name(t);
    const bool want = cap->active && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
        (strncmp(name, "kq-", 3) == 0 || strncmp(name, "indexer_kq-", 11) == 0 || strncmp(name, "indexer_score-", 14) == 0);
    if (ask) {
        return want;
    }
    if (want) {
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
        auto it = std::find_if(cap->data.begin(), cap->data.end(), [&](const auto & e) { return e.first == name; });
        if (it == cap->data.end()) {
            cap->data.emplace_back(name, std::move(buf));
        } else {
            it->second = std::move(buf);
        }
    }
    return true;
}

struct kq_diff {
    double      worst = 0.0;
    std::string first_bad; // first tensor above the threshold, later layers can inherit its error through Q
    std::string bad;       // all tensors above the threshold, with their nmse
};

static kq_diff compare_kq(const std::vector<std::pair<std::string, std::vector<float>>> & ref,
                          const std::vector<std::pair<std::string, std::vector<float>>> & cur, double thold) {
    if (ref.size() != cur.size()) {
        throw std::runtime_error("kq tensor count mismatch");
    }
    kq_diff res;
    for (size_t i = 0; i < ref.size(); i++) {
        if (ref[i].first != cur[i].first || ref[i].second.size() != cur[i].second.size()) {
            throw std::runtime_error("kq tensor mismatch: " + ref[i].first);
        }
        double mse_a_b = 0.0;
        double mse_a_0 = 0.0;
        for (size_t j = 0; j < ref[i].second.size(); j++) {
            const double a = ref[i].second[j];
            const double b = cur[i].second[j];
            // masked rows may hold -inf in both
            if (!std::isfinite(a) && a == b) {
                continue;
            }
            mse_a_b += (a - b)*(a - b);
            mse_a_0 += a*a;
        }
        const double val = mse_a_b == 0.0 ? 0.0 : mse_a_0 == 0.0 ? INFINITY : mse_a_b/mse_a_0;
        res.worst = std::max(res.worst, std::isnan(val) ? INFINITY : val);
        if (!(val <= thold)) {
            if (res.first_bad.empty()) {
                res.first_bad = ref[i].first;
            }
            char buf[64];
            snprintf(buf, sizeof(buf), " %s=%.1e", ref[i].first.c_str(), val);
            res.bad += buf;
        }
    }
    return res;
}

// decodes the prefix at pos_prefix, applies each step of moves with its own K-shift and returns the logits of the probe at pos_probe
static std::vector<float> get_logits_kv_shift(
        llama_model * model, llama_context * lctx, kq_capture & cap, const std::vector<llama_token> & prefix, const std::vector<llama_token> & probe,
        llama_pos pos_prefix, const std::vector<std::vector<kv_shift_move>> & steps, llama_pos pos_probe) {
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_memory_t mem = llama_get_memory(lctx);

    cap.data.clear();
    llama_memory_clear(mem, true);
    decode_range(lctx, prefix.data(), prefix.size(), pos_prefix, 0, false);

    for (size_t i = 0; i < steps.size(); i++) {
        for (const kv_shift_move & mv : steps[i]) {
            llama_memory_seq_add(mem, 0, mv.p0, mv.p1, mv.delta);
        }
        // there is no public API to apply a pending shift, a decode of a throwaway token in seq 1 does it
        if (i + 1 < steps.size()) {
            decode_range(lctx, probe.data(), 1, 0, 1, false);
            llama_memory_seq_rm(mem, 1, -1, -1);
        }
    }

    cap.active = true;
    decode_range(lctx, probe.data(), probe.size(), pos_probe, 0, true);
    cap.active = false;

    std::vector<float> ret;
    ret.reserve(probe.size()*n_vocab);
    for (uint32_t i = 0; i < probe.size(); i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        ret.insert(ret.end(), logits_ith, logits_ith + n_vocab);
    }
    return ret;
}

// a K-shift is correct if decoding at [d, d+n) and shifting by -d gives the same probe logits as decoding at [0, n)
static int test_kv_shift(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    // n_prefix + n_probe stays below the fixture SWA window (n_ctx/8 = 32)
    const uint32_t  n_prefix = 20;
    const uint32_t  n_probe  = 8;
    const llama_pos d        = 64;
    const llama_pos gap      = 4;

    const std::vector<llama_token> tokens = get_tokens(n_prefix + n_probe, 128, seed);
    const std::vector<llama_token> prefix(tokens.begin(), tokens.begin() + n_prefix);
    const std::vector<llama_token> probe (tokens.begin() + n_prefix, tokens.end());

    const std::vector<std::vector<kv_shift_move>> steps_shift  = { { { d, d + (llama_pos) n_prefix, -d } } };
    const std::vector<std::vector<kv_shift_move>> steps_shift2 = { { { d, d + (llama_pos) n_prefix, -d/2 } }, { { d/2, d/2 + (llama_pos) n_prefix, -d/2 } } };
    // control: the second half of the prefix ends up gap positions too far, which tells if the probe is sensitive to K positions at all
    const std::vector<std::vector<kv_shift_move>> steps_ctrl   = { {
        { d,                           d + (llama_pos) n_prefix/2, -d       },
        { d + (llama_pos) n_prefix/2,  d + (llama_pos) n_prefix,   -d + gap },
    } };

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llama_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llama_arch_name(arch)));
    }
    const std::string template_row = "|%" + std::to_string(max_arch_name_length) + "s|%6s|%10s|%10s|%10s|%10s|%s\n";

    size_t n_tests  = 0;
    size_t n_failed = 0;
    common_log_flush(common_log_main());
    LOG(template_row.c_str(), "Model arch.", "Config", "Shift kq", "Shift2 kq", "Logits", "Control kq", "Status");

    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN || !arch_matches(arch_filter, arch) || !arch_supported(arch)) {
            continue;
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue;
        }
        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        if (encode) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const char * config_name = moe ? "MoE" : "Dense";

            // the attention scale depends on the absolute position, so the hidden states of the prefix do too
            const double thold = arch == LLM_ARCH_MISTRAL_LARGE4 ? 1e-4 : 1e-6;

            // at least 4 layers, so that "every 4th layer" patterns (NoPE, full attention) are covered
            // exaone4 is SWA only with 64 layers, smallthinker always ropes with 4
            const uint32_t n_layer_min = arch == LLM_ARCH_EXAONE4 ? 64 : arch == LLM_ARCH_SMALLTHINKER ? 8 : 4;
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe, n_layer_min);
            // different bases, so that a shift using the wrong one is caught
            gguf_set_val_f32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_ROPE_FREQ_BASE).c_str(),     20000.0f);
            gguf_set_val_f32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_ROPE_FREQ_BASE_SWA).c_str(), 10000.0f);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }

            llama_model_ptr model;
            try {
                model = get_model(gguf_ctx.get(), nullptr, seed, stdev, {});
            } catch (const std::exception & err) {
                LOG(template_row.c_str(), llama_arch_name(arch), config_name, "", "", "", "", "\033[1;31mERROR\033[0m (model load)");
                n_tests++;
                n_failed++;
                continue;
            }
            // absolute position embeddings are not translation invariant, the comparison does not hold
            if (llama_model_rope_type(model.get()) == LLAMA_ROPE_TYPE_NONE) {
                LOG(template_row.c_str(), llama_arch_name(arch), config_name, "", "", "", "", "\033[1;33mSKIP\033[0m (no rope)");
                continue;
            }

            LOG("|%*s|%6s|", (int) max_arch_name_length, llama_arch_name(arch), config_name);
            fflush(stdout);

            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx           = 0;
            ctx_params.n_threads       = 4;
            ctx_params.n_threads_batch = 4;
            ctx_params.n_seq_max       = 2;
            ctx_params.kv_unified      = true;
            ctx_params.type_k          = GGML_TYPE_F32;
            ctx_params.type_v          = GGML_TYPE_F32;
            ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            // differs from the SWA default, so that a shift using the wrong scale is caught
            ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
            ctx_params.rope_freq_scale   = 0.5f;
            kq_capture cap;
            ctx_params.cb_eval           = kq_capture_cb;
            ctx_params.cb_eval_user_data = &cap;

            ud.verbosity = LOG_LEVEL_OUTPUT;
            llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
            ud.verbosity = verbosity;
            if (!lctx) {
                LOG("%10s|%10s|%10s|%10s|\033[1;33mSKIP\033[0m (context)\n", "", "", "", "");
                continue;
            }
            llama_memory_t mem = llama_get_memory(lctx.get());
            if (mem == nullptr || !llama_memory_can_shift(mem)) {
                LOG("%10s|%10s|%10s|%10s|\033[1;33mSKIP\033[0m (no shift)\n", "", "", "", "");
                continue;
            }

            std::string status;
            char shift_str[16]  = "";
            char shift2_str[16] = "";
            char logits_str[16] = "";
            char ctrl_str[16]   = "";
            n_tests++;
            try {
                const std::vector<float> logits_ref = get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, 0, {}, n_prefix);
                const auto kq_ref = cap.data;
                if (verbosity >= LOG_LEVEL_INFO) {
                    for (const auto & e : kq_ref) {
                        LOG("[%s: %zu] ", e.first.c_str(), e.second.size());
                    }
                }
                const std::vector<float> logits_shift = get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, d, steps_shift, n_prefix);
                const kq_diff diff_shift = compare_kq(kq_ref, cap.data, thold);
                get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, d, steps_shift2, n_prefix);
                const kq_diff diff_shift2 = compare_kq(kq_ref, cap.data, thold);
                const std::vector<float> logits_ctrl = get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, d, steps_ctrl, n_prefix + gap);
                const kq_diff diff_ctrl = compare_kq(kq_ref, cap.data, thold);

                const double nmse_logits = nmse(logits_ref, logits_shift);
                snprintf(shift_str,  sizeof(shift_str),  "%.2e", diff_shift.worst);
                snprintf(shift2_str, sizeof(shift2_str), "%.2e", diff_shift2.worst);
                snprintf(logits_str, sizeof(logits_str), "%.2e", nmse_logits);
                snprintf(ctrl_str,   sizeof(ctrl_str),   "%.2e", diff_ctrl.worst);

                if (kq_ref.empty()) {
                    status = "\033[1;33mSKIP\033[0m (no kq captured)";
                } else if (!diff_shift.first_bad.empty() || !diff_shift2.first_bad.empty()) {
                    n_failed++;
                    status = "\033[1;31mFAIL\033[0m" + (diff_shift.bad.empty() ? " x2:" + diff_shift2.bad : diff_shift.bad);
                } else if (!(diff_ctrl.worst > 10*thold)) {
                    status = "\033[1;33mOK\033[0m (low sensitivity)";
                } else {
                    status = "\033[1;32mOK\033[0m";
                }
            } catch (const std::exception & err) {
                n_failed++;
                status = std::string("\033[1;31mERROR\033[0m (") + err.what() + ")";
            }
            LOG("%10s|%10s|%10s|%10s|%s\n", shift_str, shift2_str, logits_str, ctrl_str, status.c_str());
        }
    }

    if (n_tests == 0) {
        LOG("Summary: no tests executed\n");
    } else if (n_failed == 0) {
        LOG("Summary: all %zu test(s) passed\n", n_tests);
    } else {
        LOG("Summary: %zu test(s) executed, %zu failed\n", n_tests, n_failed);
    }

    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return n_failed == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    // init the logger at max verbosity. filter with a custom callback respecting the user-configure verbosity
    common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);
    common_init();
    ggml_backend_load_all();
    llama_backend_init();

    std::random_device rd;

    std::string arch_filter;
    size_t seed = rd();
    float stdev = 0.1f;
    std::string out;
    const char * target_backend = nullptr;
    bool graph_recovery = false;
    bool layer_inputs = false;
    bool dflash_device_inputs = false;
    bool dflash_selector = false;
    bool dflash_vision = false;
    bool dflash_tail = false;
    bool kq_derived = false;
    bool qsa_sparse = false;
    bool moe_cache_policy = false;
    bool moe_cache_prefill = false;
    bool kv_shift = false;

    int verbosity = LOG_LEVEL_ERROR;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                if (llama_arch_from_string(arch_name) != LLM_ARCH_UNKNOWN) {
                    // exact architecture name
                    arch_filter = "^" + arch_name + "$";
                } else {
                    try {
                        std::regex re(arch_name);
                        arch_filter = arch_name;
                    } catch (const std::regex_error & err) {
                        LOG_ERR("%s: invalid architecture regex: %s (%s)\n", __func__, arch_name.c_str(), err.what());
                        return 1;
                    }
                }
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--stdev") == 0) {
            if (i + 1 < argc) {
                stdev = std::stof(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            if (i + 1 < argc) {
                verbosity = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--backend") == 0) {
            if (i + 1 < argc) {
                const char * backend_name = argv[++i];
                ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name);
                if (dev == nullptr) {
                    LOG_ERR("%s: unknown backend device: %s\n", __func__, backend_name);
                    return 1;
                }
                target_backend = ggml_backend_dev_name(dev);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "--test-graph-recovery") == 0) {
            graph_recovery = true;
        } else if (strcmp(argv[i], "--test-layer-inputs") == 0) {
            layer_inputs = true;
        } else if (strcmp(argv[i], "--test-dflash-device-inputs") == 0) {
            dflash_device_inputs = true;
        } else if (strcmp(argv[i], "--test-dflash-selector") == 0) {
            dflash_selector = true;
        } else if (strcmp(argv[i], "--test-dflash-vision") == 0) {
            dflash_vision = true;
        } else if (strcmp(argv[i], "--test-moe-cache-prefill") == 0) {
            moe_cache_prefill = true;
        } else if (strcmp(argv[i], "--test-moe-cache-policy") == 0) {
            moe_cache_policy = true;
        } else if (strcmp(argv[i], "--test-qsa-sparse") == 0) {
            qsa_sparse = true;
        } else if (strcmp(argv[i], "--test-kq-derived") == 0) {
            kq_derived = true;
        } else if (strcmp(argv[i], "--test-dflash-tail") == 0) {
            dflash_tail = true;
        } else if (strcmp(argv[i], "--kv-shift") == 0) {
            kv_shift = true;
        } else {
            LOG_ERR("%s: unknown argument: %s\n", __func__, argv[i]);
            usage(argv);
            return 1;
        }
    }
    if (stdev <= 0.0f) {
        LOG_ERR("%s: stdev must be > 0\n", __func__);
        return 1;
    }
    LOG_INF("%s: using seed %zu, stdev %f\n", __func__, seed, stdev);

    init_normal_pool(seed, stdev);

    try {
#ifdef LLAMA_TEST_FORK_INTERNALS
        if (moe_cache_prefill) {
            return test_moe_cache_prefill(seed, stdev, target_backend);
        }
        if (moe_cache_policy) {
            return test_moe_cache_policy(seed);
        }
        if (qsa_sparse) {
            return test_qsa_sparse(seed, stdev);
        }
        if (kq_derived) {
            return test_kq_derived(seed, stdev);
        }
        if (dflash_tail) {
            const char * tail_env = getenv("LLAMA_DFLASH_INGEST_TAIL");
            const int n_modes = tail_env && atoi(tail_env) == 0 ? 1 : 3;
            for (int mode = 0; mode < n_modes; ++mode) {
                if (test_dflash_vision(seed, stdev, mode) != 0) {
                    return 1;
                }
            }
            return 0;
        }
        if (dflash_device_inputs) {
            if (test_dflash_device_inputs(seed, stdev, target_backend, 1) != 0) {
                return 1;
            }
            return test_dflash_device_inputs(seed, stdev, target_backend, 2);
        }
        if (dflash_selector) {
            return test_dflash_selector(seed, stdev, target_backend);
        }
        if (dflash_vision) {
            return test_dflash_vision(seed, stdev);
        }
        if (layer_inputs) {
            return test_layer_inputs(seed, stdev);
        }
        if (graph_recovery) {
            return test_graph_recovery(seed, stdev);
        }
#else
        if (moe_cache_prefill || moe_cache_policy || qsa_sparse || kq_derived || dflash_tail ||
                dflash_device_inputs || dflash_selector || dflash_vision || layer_inputs || graph_recovery) {
            LOG_ERR("%s: fork-internal tests require access to internal symbols\n", __func__);
            return 1;
        }
#endif
        if (!out.empty()) {
            return save_models(arch_filter, seed, stdev, verbosity, out);
        }
        if (kv_shift) {
            return test_kv_shift(arch_filter, seed, stdev, verbosity);
        }
        return test_backends(arch_filter, seed, stdev, verbosity, target_backend);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
