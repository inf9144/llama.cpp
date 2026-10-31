#include "llama-moe-measure.h"
#include "llama-moe-cache.h"
#include "llama-moe-cache-policy.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-batch.h"
#include "../ggml/src/ggml-moe.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <locale>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <filesystem>
#endif

llama_moe_cache_link llama_moe_cache_host_link(const char * device_id, const char * root) {
#if defined(__linux__)
    auto pci_id = [](const char * id) -> std::string {
        unsigned domain, bus, device, function;
        char extra;
        if (!id || sscanf(id, "%4x:%2x:%2x.%1x%c", &domain, &bus, &device, &function, &extra) != 4 ||
                domain > 0xffff || bus > 0xff || device > 0x1f || function > 7) return {};
        char bdf[16];
        snprintf(bdf, sizeof(bdf), "%04x:%02x:%02x.%x", domain, bus, device, function);
        return bdf;
    };
    const auto bdf = pci_id(device_id);
    if (bdf.empty()) return {};

    std::error_code ec;
    auto path = std::filesystem::canonical(std::filesystem::path(root) / bdf, ec);
    if (ec) return {};
    llama_moe_cache_link result;
    for (; !pci_id(path.filename().string().c_str()).empty(); path = path.parent_path()) {
        double speed = 0;
        int width = 0;
        std::string unit;
        std::ifstream speed_file(path / "max_link_speed");
        std::ifstream width_file(path / "current_link_width");
        speed_file.imbue(std::locale::classic());
        width_file.imbue(std::locale::classic());
        // Ignore idle speed changes, but retain electrically negotiated lanes.
        if (!(speed_file >> speed >> unit) || !(width_file >> width) || unit != "GT/s" ||
                !(speed >= 2.5 && speed <= 128 && width > 0 && width <= 32)) return {};
        if (result.bandwidth() == 0 || speed*width < result.bandwidth()) {
            result = { speed, width, path.filename().string() };
        }
    }
    return result;
#else
    GGML_UNUSED(device_id);
    GGML_UNUSED(root);
    return {};
#endif
}

namespace {

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget
        size_t budget = 0;     // explicit per-device budget
        bool prefill = false;
        bool device_maps = false;
        llama_moe_cache_link host_link;
        size_t stage_stride = 0;
        size_t stage_span = 0;
        int32_t max_experts = 0;
        ggml_tensor * stage = nullptr;
        ggml_tensor * addresses = nullptr;
        ggml_tensor * active = nullptr;
        ggml_moe_pipeline pipeline = {};

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
        size_t scratch_size = 0;
    };

    // Layers of one device/layout share banks and window-based admission history.
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_policy policy;
        bool streaming = false;
        int32_t prefill_id = -1;
        ggml_tensor * gpu_map = nullptr;
        std::vector<int32_t> gpu_map_host;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        ggml_tensor * slot_map_prefill = nullptr; // aliases slot_map; identifies streaming execution
        ggml_tensor * slot_map_device = nullptr;
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
        std::vector<int32_t> selected;
        std::vector<int32_t> temporary;
        std::vector<int32_t> counts;
    };

    struct binding {
        impl * owner;
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
        ggml_tensor * prefill_cached;
        struct metadata {
            std::vector<uint64_t> addresses;
            std::vector<int32_t> experts;
        };
        std::vector<metadata> packets; // host upload sources live through the next routing sync
        ggml_prec activation_precision = GGML_PREC_DEFAULT;
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    int32_t n_expert_used;

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches
    stats stats_prefill;

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    struct selection {
        int32_t il;
        bool prefill;
    };
    std::unordered_map<const ggml_tensor *, selection> layer_of; // slot map -> layer and admission policy

    std::vector<int32_t> ids;
    std::vector<moe_cache_policy::fill> fills;
    moe_cache_clock clock;
    std::mutex stats_mutex;
    int32_t batch_tokens = 0;
    bool pending_compute = false;
#ifdef LLAMA_MOE_MEASURE
    uint64_t measure_graph = 0;
    uintptr_t measure_context = 0;
    int measure_type = 0;
#endif

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, const size_t * sizes) :
            n_expert_used(model.hparams.n_expert_used_max()), layers(model.layers.size()) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
                auto dev = ggml_backend_get_device(d.backend);
                auto reg = ggml_backend_dev_backend_reg(dev);
                using supports_prefill_fn = bool (*)(ggml_backend_dev_t);
                auto supports = reg ? (supports_prefill_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hip_moe_cache_supports_prefill") : nullptr;
                d.prefill = supports && supports(dev);
                if (d.prefill) {
                    ggml_backend_dev_props props = {};
                    ggml_backend_dev_get_props(dev, &props);
                    d.host_link = llama_moe_cache_host_link(props.device_id);
                    if (d.host_link.bandwidth() > 0) {
                        LLAMA_LOG_INFO("%s: %10s host link = %.1f GT/s x%d, limiting device %s (GPU %s)\n", __func__,
                            ggml_backend_name(d.backend), d.host_link.speed, d.host_link.width,
                            d.host_link.limiting_device.c_str(), props.device_id);
                    } else {
                        LLAMA_LOG_WARN("%s: %10s host link = unknown (GPU %s); keep local prefill\n", __func__,
                            ggml_backend_name(d.backend), props.device_id ? props.device_id : "unknown");
                    }
                }
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        // if sizes is set, each device gets that budget
        const bool per_device = sizes != nullptr;
        if (per_device) {
            for (size_t i = 0; i < model.n_devices(); ++i) {
                if (sizes[i] == 0) {
                    continue;
                }
                const auto it = std::find_if(devices.begin(), devices.end(), [&](const device & d) {
                    return ggml_backend_get_device(d.backend) == model.devices[i].dev;
                });
                if (it == devices.end() || it->host_bytes == 0) {
                    LLAMA_LOG_WARN("%s: unused MoE cache on device %zu, no host experts\n", __func__, i);
                }
            }
        }
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = !per_device && (tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; }));
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (per_device || !split_by_free) {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                const size_t id_model = (size_t) (it - model.devices.begin());
                if (per_device) {
                    d.budget = sizes[id_model];
                } else {
                    d.split = (double) tensor_split[id_model];
                }
            } else {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            }
            split_sum += d.split;
        }
        if (!per_device && split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            g.prefill_id = g.id;
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget = per_device
                ? (size_t) ((double) d.budget*g.host_bytes/d.host_bytes)
                : (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.policy.init(model.layers.size(), n_expert, g.n_slots);
            g.streaming = d.prefill && n_expert <= 32768 && std::all_of(g.ref.begin(), g.ref.end(), [](const ggml_tensor * w) {
                return ggml_moe_stream_weight_supported(w);
            });
            for (int32_t il : g.layers) {
                for (const auto * w : llama_moe_cache_layer_experts(model.layers[il])) {
                    const auto it = model.prec_policy.prec_src1.find(w);
                    if (it != model.prec_policy.prec_src1.end() && it->second != GGML_PREC_DEFAULT && it->second != GGML_PREC_Q8) {
                        g.streaming = false;
                    }
                }
            }
            if (g.streaming) {
                // Preserve locality unless the owner's uncached fraction costs
                // more link time than full host streaming on a faster device.
                if (n_expert_used <= moe_cache_stage_slots) {
                    for (size_t id = 0; id < devices.size(); ++id) {
                        const auto & candidate = devices[id];
                        if (candidate.prefill && candidate.host_link.bandwidth() > devices[g.prefill_id].host_link.bandwidth() &&
                                moe_cache_stage_on_faster_link(d.host_link.bandwidth(), candidate.host_link.bandwidth(), g.n_slots, max_slots)) {
                            g.prefill_id = id;
                        }
                    }
                }
                auto & device = devices[g.prefill_id];
                device.max_experts = std::max(device.max_experts, n_expert);
                for (const auto * w : g.ref) device.stage_stride = std::max(device.stage_stride, w->nb[2]);
                if (g.prefill_id != g.id) n_tensors[g.prefill_id] += g.ref.size()*g.layers.size();
            }
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            if (g.streaming) n_tensors[g.id] += 1 + g.layers.size();
            n_tensors_host  += 2*g.layers.size();
        }
        if (n_tensors_host == 0) {
            if (per_device) {
                LLAMA_LOG_WARN("%s: no device can use the per-device MoE cache, MoE cache is disabled\n", __func__);
                return;
            }
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                auto & d = devices[id];
                d.ctx = init_ctx(n_tensors[id] + (d.stage_stride ? 3 : 0));
                if (d.stage_stride) {
                    const size_t alignment = ggml_backend_buft_get_alignment(d.buft);
                    d.stage_span = GGML_PAD((moe_cache_stage_slots + 1)*d.stage_stride, alignment);
                    d.stage = ggml_new_tensor_1d(d.ctx.get(), GGML_TYPE_I8, 2*d.stage_span);
                    d.addresses = ggml_new_tensor_1d(d.ctx.get(), GGML_TYPE_I64, 2*d.max_experts);
                    d.active = ggml_new_tensor_1d(d.ctx.get(), GGML_TYPE_I32, 2*d.max_experts);
                    ggml_set_name(d.stage, "ffn_moe_miss_stage");
                    ggml_set_name(d.addresses, "moe_cache.addresses");
                    ggml_set_name(d.active, "moe_cache.active");
                    d.scratch_size = GGML_PAD(ggml_nbytes(d.stage), alignment) +
                        GGML_PAD(ggml_nbytes(d.addresses), alignment) + GGML_PAD(ggml_nbytes(d.active), alignment);
                    d.buf_size += d.scratch_size;
                }
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            if (g.streaming) {
                const int64_t entries = int64_t(g.layers.size())*g.ref[0]->ne[2];
                g.gpu_map = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, entries);
                ggml_format_name(g.gpu_map, "moe_cache.device_maps-%zu", ig);
                g.gpu_map_host.assign(entries, -1);
                devices[g.id].device_maps = true;
                devices[g.id].buf_size += GGML_PAD(ggml_nbytes(g.gpu_map), ggml_backend_buft_get_alignment(devices[g.id].buft));
            }
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    ggml_tensor * prefill_cached = cached;
                    if (g.streaming && g.prefill_id != g.id) {
                        auto & execution = devices[g.prefill_id];
                        prefill_cached = ggml_new_tensor_3d(execution.ctx.get(), bank->type, bank->ne[0], bank->ne[1], moe_cache_stage_slots);
                        prefill_cached->view_src = execution.stage;
                        GGML_ASSERT(ggml_nbytes(prefill_cached) <= ggml_nbytes(execution.stage));
                        ggml_format_name(prefill_cached, "moe_cache.stream.%s", l.experts[ip]->name);
                    }
                    bindings[l.experts[ip]] = { this, il, (int32_t) ip, cached, prefill_cached, {}, GGML_PREC_DEFAULT };
                    const auto precision = model.prec_policy.prec_src1.find(l.experts[ip]);
                    if (precision != model.prec_policy.prec_src1.end()) bindings[l.experts[ip]].activation_precision = precision->second;
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                l.slot_map_prefill = ggml_view_2d(ctx_host.get(), l.slot_map, 1, l.slot_map->ne[1], l.slot_map->nb[1], 0);
                ggml_format_name(l.slot_map_prefill, "moe_cache.slot_map_prefill-%d", il);
                if (g.gpu_map) {
                    const size_t index = std::find(g.layers.begin(), g.layers.end(), il) - g.layers.begin();
                    l.slot_map_device = ggml_view_2d(ctx, g.gpu_map, 1, l.slot_map->ne[1], sizeof(int32_t), index*l.slot_map->ne[1]*sizeof(int32_t));
                    ggml_format_name(l.slot_map_device, "moe_cache.device_map-%d", il);
                }
                layer_of[l.slot_map] = { il, false };
                layer_of[l.slot_map_prefill] = { il, true };
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                if (g.gpu_map) ggml_backend_tensor_memset(g.gpu_map, 0xff, 0, ggml_nbytes(g.gpu_map));
                for (int32_t il : g.layers) {
                    if (layers[il].slot_map != nullptr) {
                        g.policy.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        if (!model.hparams.no_alloc) {
            for (auto & d : devices) {
                if (!d.prefill || !d.ctx || (!d.stage && !d.device_maps)) continue;
                auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(d.backend));
                auto init = (ggml_moe_pipeline_init_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hip_moe_pipeline_init");
                if (init) init(d.backend, layers.size(), &impl::resolve, this, &d.pipeline);
                LLAMA_LOG_INFO("%s: %10s device routing = %s, transfer pipeline = %s\n", __func__,
                    ggml_backend_name(d.backend), d.pipeline.device_lookup ? "enabled" : "fallback",
                    d.pipeline.context ? "enabled" : "fallback");
            }
        }

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0 && !d.scratch_size) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), (d.buf_size-d.scratch_size)/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            if (d.scratch_size) {
                LLAMA_LOG_INFO("%s: %10s bounded miss scratch = %.2f MiB (2 x %d experts + guard, shared across banks)\n", __func__,
                    ggml_backend_buft_name(d.buft), d.scratch_size/1024.0/1024.0, moe_cache_stage_slots);
            }
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
                }
            }
            for (size_t execution = 0; execution < devices.size(); ++execution) {
                if (std::any_of(groups.begin(), groups.end(), [&](const group & g) {
                        return g.id == (int32_t) id && g.streaming && g.prefill_id == (int32_t) execution;
                    })) {
                    LLAMA_LOG_INFO("%s: %10s prefill execution = %s\n", __func__,
                        ggml_backend_name(d.backend), ggml_backend_name(devices[execution].backend));
                }
            }
        }
    }

    ggml_backend_t backend(int32_t il, int64_t n_tokens) const {
        const group & g = groups[layers[il].ig];
        return devices[n_tokens > moe_cache_max_batch && g.streaming ? g.prefill_id : g.id].backend;
    }

    ~impl() {
        synchronize();
        for (auto & d : devices) {
            if (d.pipeline.context) d.pipeline.destroy(d.pipeline.context);
        }
        log_stats();
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        const group & g = groups[l.ig];
        const bool prefill = n_tokens > moe_cache_max_batch;
        // Streaming has a fixed miss bank: it does not need all selected experts
        // to fit the resident cache. Legacy small batches retain their MMVQ path.
        if (n_expert_used <= 0 || (prefill ? (!g.streaming || n_tokens > 65535 / n_expert_used) :
                !moe_cache_fits(n_tokens, n_expert_used, l.slot_map->ne[1], g.n_slots))) {
            return nullptr;
        }
        if (!prefill && n_tokens <= 8 && n_expert_used*n_tokens <= 256 && g.streaming && devices[g.id].pipeline.device_lookup) {
            return l.slot_map_device;
        }
        return prefill ? l.slot_map_prefill : l.slot_map;
    }

    ggml_tensor * lookup_slots(ggml_context * ctx, ggml_tensor * map, ggml_tensor * ids, int32_t il) const {
        const auto & l = layers[il];
        if (map == l.slot_map_device) {
            const auto & p = devices[groups[l.ig].id].pipeline;
            return ggml_moe_cache_lookup(ctx, map, ids, p.lookup_job(p.context, il));
        }
        return ggml_get_rows(ctx, map, ids);
    }

    void synchronize() {
        if (!pending_compute) return;
        for (auto & d : devices) if (d.pipeline.context) d.pipeline.flush(d.pipeline.context);
        pending_compute = false;
    }

    void begin_compute(ggml_cgraph * graph) {
        synchronize();
        MOE_MEASURE(measure_graph = llama_moe_measure::graph; measure_context = llama_moe_measure::context; measure_type = llama_moe_measure::graph_type);
        bool needs_lookup = batch_tokens <= 8;
        if (!needs_lookup) {
            // A large ubatch can crop its last MoE layer to a small output batch.
            const int n_nodes = ggml_graph_n_nodes(graph);
            for (int i = 0; i < n_nodes; ++i) {
                if (ggml_graph_node(graph, i)->op == GGML_OP_MOE_CACHE_LOOKUP) {
                    needs_lookup = true;
                    break;
                }
            }
        }
        if (needs_lookup) {
            for (auto & d : devices) if (d.device_maps && d.pipeline.context) d.pipeline.begin(d.pipeline.context);
        }
        pending_compute = true;
    }

    static void resolve(void * userdata, int32_t il, const int32_t * ids, int32_t count, bool miss) {
        ((impl *) userdata)->resolve(il, ids, count, miss);
    }

    void resolve(int32_t il, const int32_t * selected, int32_t count, bool miss) {
        auto & g = groups[layers[il].ig];
        auto & d = devices[g.id];
        MOE_MEASURE(llama_moe_measure::graph = measure_graph; llama_moe_measure::context = measure_context;
            llama_moe_measure::tokens = batch_tokens; llama_moe_measure::graph_type = measure_type);
        MOE_MEASURE(std::vector<int32_t> resident(g.policy.n_expert, -1);
            if (miss) std::copy(g.policy.slot_map[il], g.policy.slot_map[il] + g.policy.n_expert, resident.begin());
            else for (int i = 0; i < count; ++i) resident[selected[i]] = 0;
            llama_moe_measure::routing(d.backend, layers[il].experts.front(), selected, count, g.policy.n_expert, 0, resident.data()));
        std::vector<moe_cache_policy::fill> admissions;
        size_t hits = 0;
#ifdef LLAMA_MOE_MEASURE
        const int64_t start = ggml_time_us();
#endif
        // Hit observations can lag behind execution. They update demand only;
        // a later miss may already have replaced a slot used by that past hit.
        GGML_ASSERT(g.policy.plan(il, selected, count, admissions, hits, clock.tokens, true, clock.latency_tokens, miss));
        if (!miss) hits = g.policy.uniq.size();
        MOE_MEASURE(llama_moe_measure::record r("hybrid_plan", d.backend, layers[il].experts.front());
            r.out << ",\"host_plan_us\":" << ggml_time_us()-start << ",\"resident_hits\":" << hits
                  << ",\"admissions\":" << admissions.size() << ",\"temporary\":0,\"device_lookup\":true,\"device_miss\":" << (miss ? "true" : "false"));
        size_t bytes = 0;
        for (size_t ip = 0; ip < layers[il].experts.size(); ++ip) {
            const auto * w = layers[il].experts[ip];
            MOE_MEASURE(llama_moe_measure::record r("bank", d.backend, w); r.out << ",\"expert_bytes\":" << w->nb[2]);
            for (size_t i = 0; i < admissions.size();) {
                size_t n = 1;
                while (i + n < admissions.size() && admissions[i+n].expert == admissions[i].expert + int32_t(n) && admissions[i+n].slot == admissions[i].slot + int32_t(n)) ++n;
#ifdef LLAMA_MOE_MEASURE
                const int64_t upload_start = ggml_time_us();
#endif
                d.pipeline.upload(d.pipeline.context, g.banks[ip], (const char *) w->data + size_t(admissions[i].expert)*w->nb[2], size_t(admissions[i].slot)*w->nb[2], n*w->nb[2]);
                MOE_MEASURE(llama_moe_measure::transfer("admission_h2d", d.backend, w, admissions[i].expert, n, n*w->nb[2], ggml_time_us()-upload_start));
                bytes += n*w->nb[2];
                i += n;
            }
        }
        if (miss) {
            for (size_t i = 0; i < g.layers.size(); ++i) {
                std::copy(g.policy.slot_map[g.layers[i]], g.policy.slot_map[g.layers[i]] + g.policy.n_expert, g.gpu_map_host.begin() + i*g.policy.n_expert);
            }
            d.pipeline.upload(d.pipeline.context, g.gpu_map, g.gpu_map_host.data(), 0, ggml_nbytes(g.gpu_map));
        }
        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            stats_small.hits += hits;
            stats_small.misses += admissions.size();
            stats_small.bytes += bytes;
        }
        MOE_MEASURE(llama_moe_measure::flush());
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.policy.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second.il;
        const bool prefill = it->second.prefill;
        layer & l = layers[il];
        group & g = groups[l.ig];

        const bool local = !prefill || g.prefill_id == g.id;
        GGML_ASSERT(backend == devices[prefill ? g.prefill_id : g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        ids.resize(ggml_nelements(sel));
#ifdef LLAMA_MOE_MEASURE
        const int64_t routing_start = ggml_time_us();
#endif
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        MOE_MEASURE(llama_moe_measure::routing(backend, l.experts.front(), ids.data(), ids.size(), l.slot_map->ne[1], ggml_time_us()-routing_start,
            local ? g.policy.slot_map[il] : nullptr));

#ifdef LLAMA_MOE_MEASURE
        const int64_t planning_start = ggml_time_us();
#endif
        size_t n_hit = 0;
        if (!g.policy.plan(il, ids.data(), ids.size(), fills, n_hit, clock.tokens, !prefill, clock.latency_tokens, local)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }

        if (local && g.gpu_map && !fills.empty()) {
            for (size_t i = 0; i < g.layers.size(); ++i) {
                std::copy(g.policy.slot_map[g.layers[i]], g.policy.slot_map[g.layers[i]] + g.policy.n_expert, g.gpu_map_host.begin() + i*g.policy.n_expert);
            }
            ggml_backend_tensor_set_async(backend, g.gpu_map, g.gpu_map_host.data(), 0, ggml_nbytes(g.gpu_map));
        }

        MOE_MEASURE(llama_moe_measure::record r("hybrid_plan", backend, l.experts.front());
            r.out << ",\"host_plan_us\":" << ggml_time_us()-planning_start
                  << ",\"resident_hits\":" << n_hit << ",\"admissions\":" << fills.size()
                  << ",\"temporary\":" << g.policy.temporary.size() << ",\"clock\":" << clock.tokens
                  << ",\"latency_clock\":" << clock.latency_tokens << ",\"local_cache\":" << (local ? "true" : "false"));
        l.selected = g.policy.uniq;
        l.temporary = g.policy.temporary;
        l.counts.assign(g.policy.counts.begin(), g.policy.counts.end());
        if (prefill) {
            for (auto * w : l.experts) {
                bindings.at(w).packets.resize(1 + (l.temporary.size() + moe_cache_stage_slots - 1)/moe_cache_stage_slots);
            }
        }

        // Admission uploads go straight to their final resident slots, in every
        // bank of the expert bundle. No staging-to-cache copies are needed.
        // Consecutive experts going to consecutive slots are uploaded together.
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            MOE_MEASURE(llama_moe_measure::record r("bank", backend, w); r.out << ",\"expert_bytes\":" << expert_size);
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
#ifdef LLAMA_MOE_MEASURE
                const int64_t upload_start = ggml_time_us();
#endif
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                MOE_MEASURE(llama_moe_measure::transfer("admission_h2d", backend, w, fills[i].expert, n, n*expert_size, ggml_time_us()-upload_start));
                bytes += n*expert_size;
                i += n;
            }
        }

        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            stats & st = prefill ? stats_prefill : ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
            st.hits   += n_hit;
            st.misses += fills.size() + l.temporary.size();
            st.bytes  += bytes;
        }

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    void set_batch(const llama_ubatch & batch) {
        synchronize();
        batch_tokens = batch.n_tokens;
        // M-RoPE's first position plane is the sequential token position.
        std::unordered_map<int32_t, std::pair<int64_t, int64_t>> ranges;
        for (uint32_t t = 0; t < batch.n_tokens; ++t) {
            for (int32_t i = 0; i < batch.n_seq_id[t]; ++i) {
                const int32_t seq = batch.seq_id[t][i];
                auto ins = ranges.emplace(seq, std::make_pair(batch.pos[t], batch.pos[t]));
                ins.first->second.first = std::min<int64_t>(ins.first->second.first, batch.pos[t]);
                ins.first->second.second = std::max<int64_t>(ins.first->second.second, batch.pos[t]);
            }
        }
        for (const auto & r : ranges) clock.begin(r.first, r.second.first, r.second.second, batch.n_tokens <= moe_cache_max_batch);
    }

    static bool prepare(void * userdata, int32_t packet, ggml_moe_packet * out) {
        auto & b = *static_cast<binding *>(userdata);
        return b.owner->prepare(b, packet, out);
    }

    bool prepare(binding & b, int32_t packet, ggml_moe_packet * out) {
        const layer & l = layers[b.il];
        const group & g = groups[l.ig];
        device & d = devices[g.prefill_id];
        const int lane = d.pipeline.context ? packet % 2 : 0;
        const size_t stage_offset = lane*d.stage_span;
        auto upload = [&](ggml_tensor * dst, const void * src, size_t offset, size_t bytes) {
            if (d.pipeline.context) d.pipeline.upload(d.pipeline.context, dst, src, offset, bytes);
            else ggml_backend_tensor_set_async(d.backend, dst, src, offset, bytes);
        };
        const size_t first = packet > 0 ? size_t(packet - 1)*moe_cache_stage_slots : 0;
        if (packet > 0 && first >= l.temporary.size()) return false;
        const ggml_tensor * w = l.experts[b.ip];
        auto & packet_addresses = b.packets.at(packet).addresses;
        auto & packet_experts = b.packets.at(packet).experts;
        packet_addresses.assign(w->ne[2], 0);
        packet_experts.clear();
        int32_t max_tokens = 0, rows = 0;
        if (packet == 0 && g.prefill_id == g.id) {
            for (int32_t e : l.selected) {
                const int32_t slot = g.policy.slot_map[b.il][e];
                if (slot < 0) continue;
                packet_addresses[e] = (uintptr_t) g.banks[b.ip]->data + size_t(slot)*w->nb[2];
                packet_experts.push_back(e);
                max_tokens = std::max(max_tokens, l.counts[e]);
                rows += l.counts[e];
            }
        } else if (packet > 0) {
            if (d.pipeline.context) d.pipeline.packet_begin(d.pipeline.context, lane);
            const size_t end = std::min(first + moe_cache_stage_slots, l.temporary.size());
            for (size_t i = first; i < end;) {
                const int32_t e = l.temporary[i];
                size_t n = 1;
                while (i + n < end && l.temporary[i + n] == e + int32_t(n)) ++n;
#ifdef LLAMA_MOE_MEASURE
                const int64_t start = ggml_time_us();
#endif
                upload(d.stage, (const char *) w->data + size_t(e)*w->nb[2], stage_offset + (i-first)*w->nb[2], n*w->nb[2]);
                MOE_MEASURE(llama_moe_measure::transfer("stage_h2d", d.backend, w, e, n, n*w->nb[2], ggml_time_us()-start));
                stats_prefill.bytes += n*w->nb[2];
                for (size_t j = i; j < i + n; ++j) {
                    const int32_t expert = l.temporary[j];
                    packet_addresses[expert] = (uintptr_t) d.stage->data + stage_offset + (j-first)*w->nb[2];
                    packet_experts.push_back(expert);
                    max_tokens = std::max(max_tokens, l.counts[expert]);
                    rows += l.counts[expert];
                }
                i += n;
            }
        }
        if (packet_experts.empty()) {
            *out = {};
            return true;
        }
        if (packet == 0 && d.pipeline.context) d.pipeline.packet_begin(d.pipeline.context, lane);
        // Upload metadata through the same pool / stream as the weights.
        // The host vectors are distinct for each bank/packet and stay alive until
        // the next routing synchronization, also if the optional stage-pool name
        // filter excludes metadata. Weight uploads retain the mmap workaround.
        upload(d.addresses, packet_addresses.data(), lane*d.max_experts*sizeof(uint64_t), packet_addresses.size()*sizeof(uint64_t));
        if (!packet_experts.empty()) {
            upload(d.active, packet_experts.data(), lane*d.max_experts*sizeof(int32_t), packet_experts.size()*sizeof(int32_t));
        }
        *out = { (const uint64_t *) d.addresses->data + lane*d.max_experts, (const int32_t *) d.active->data + lane*d.max_experts,
                 (int32_t) packet_experts.size(), max_tokens, rows,
                 packet > 0 ? (char *) d.stage->data + stage_offset + packet_experts.size()*w->nb[2] : nullptr,
                 packet > 0 ? w->nb[2] : 0, nullptr, nullptr };
        if (d.pipeline.context) d.pipeline.packet_end(d.pipeline.context, lane, out);
        return true;
    }

    ggml_tensor * mul_mat(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_tensor * ids, ggml_tensor * slots) {
        auto it = bindings.find(w);
        GGML_ASSERT(it != bindings.end());
        binding & b = it->second;
        ggml_tensor * res;
        if (ids->ne[1] <= moe_cache_max_batch) {
            res = ggml_mul_mat_id(ctx, b.cached, x, slots);
        } else {
            GGML_ASSERT(groups[layers[b.il].ig].streaming);
            ggml_moe_stream stream = { static_cast<ggml_moe_prepare_t>(&impl::prepare), &b, (int32_t) w->ne[2] };
            res = ggml_mul_mat_id_stream(ctx, b.prefill_cached, x, ids, slots, stream);
        }
        ggml_prec_set_src(res, b.activation_precision, 1);
        // A model-specific activation layout / precision can be unsupported even
        // when its weights qualify. Keep the host-weight operation as fallback;
        // never schedule a process-local device-address callback on the CPU.
        return ggml_backend_supports_op(backend(b.il, ids->ne[1]), res) ? res : ggml_mul_mat_id(ctx, w, x, ids);
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        log("prefill hybrid", stats_prefill);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, const size_t * sizes) :
    pimpl(new impl(model, backends, bufts, size, sizes)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il, int64_t n_tokens) const {
    return pimpl->backend(il, n_tokens);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

void llama_moe_cache::set_batch(const llama_ubatch & batch) {
    pimpl->set_batch(batch);
}

ggml_tensor * llama_moe_cache::mul_mat(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_tensor * ids, ggml_tensor * slots) const {
    return pimpl->mul_mat(ctx, w, x, ids, slots);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

ggml_tensor * llama_moe_cache::lookup_slots(ggml_context * ctx, ggml_tensor * map, ggml_tensor * ids, int32_t il) const {
    return pimpl->lookup_slots(ctx, map, ids, il);
}

void llama_moe_cache::begin_compute(ggml_cgraph * graph) {
    pimpl->begin_compute(graph);
}

void llama_moe_cache::synchronize() {
    pimpl->synchronize();
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
