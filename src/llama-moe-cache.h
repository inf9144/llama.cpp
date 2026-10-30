#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

struct llama_model;
struct llama_ubatch;

struct llama_moe_cache_link {
    double speed = 0;
    int width = 0;
    std::string limiting_device;

    double bandwidth() const { return speed*width; }
};

// Internal PCIe estimate; the root argument also permits isolated sysfs fixtures.
llama_moe_cache_link llama_moe_cache_host_link(const char * device_id, const char * root = "/sys/bus/pci/devices");

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // sizes: optional per-device budget, llama_max_devices()
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, const size_t * sizes = nullptr);
    ~llama_moe_cache();

    // Execution device; small batches stay with the resident cache owner.
    ggml_backend_t backend(int32_t il, int64_t n_tokens = 0) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    void set_batch(const llama_ubatch & batch);

    // Select the all-resident or streaming execution plan with logical IDs.
    ggml_tensor * mul_mat(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_tensor * ids, ggml_tensor * slots) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
