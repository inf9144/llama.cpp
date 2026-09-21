#pragma once

#include "server-task.h"

#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>

// binds an SSD snapshot to a specific model + KV/context configuration
// a snapshot is only a candidate when this matches the loaded model exactly
struct server_prompt_cache_ssd_fingerprint {
    // model identity
    int32_t n_layer       = 0;
    int32_t n_embd        = 0;
    int32_t n_head        = 0;
    int32_t n_head_kv     = 0;
    int32_t n_ctx_train   = 0;
    int32_t n_swa         = 0;
    int32_t n_layer_nextn = 0;
    int32_t ftype         = 0;
    uint64_t n_params     = 0;
    uint64_t model_size   = 0;
    uint64_t file_size    = 0;
    int64_t  file_mtime   = 0;

    // KV / context parameters that affect the state layout
    int32_t n_ctx        = 0;
    int32_t flash_attn   = 0;
    int32_t kv_unified   = 0;

    bool operator==(const server_prompt_cache_ssd_fingerprint & o) const {
        return n_layer == o.n_layer && n_embd == o.n_embd && n_head == o.n_head &&
               n_head_kv == o.n_head_kv && n_ctx_train == o.n_ctx_train && n_swa == o.n_swa &&
               n_layer_nextn == o.n_layer_nextn && ftype == o.ftype &&
               n_params == o.n_params && model_size == o.model_size &&
               file_size == o.file_size && file_mtime == o.file_mtime &&
               n_ctx == o.n_ctx && flash_attn == o.flash_attn && kv_unified == o.kv_unified;
    }
    bool operator!=(const server_prompt_cache_ssd_fingerprint & o) const { return !(*this == o); }
};

// disk-backed LRU of prompt cache states, used as a persistent second tier
// below the RAM prompt cache. states are keyed by a hash of the prompt tokens
struct server_prompt_cache_ssd {
    struct entry {
        std::string key;
        size_t size = 0;          // file size in bytes
        int64_t last_access_ms = 0;
        size_t n_tokens = 0;
        server_tokens tokens;     // kept in RAM for prefix matching without reading the file
    };

    server_prompt_cache_ssd(std::string dir, size_t limit_size);
    ~server_prompt_cache_ssd();

    server_prompt_cache_ssd(const server_prompt_cache_ssd &) = delete;
    server_prompt_cache_ssd & operator=(const server_prompt_cache_ssd &) = delete;

    // scan the directory and build the in-memory index
    // entries whose fingerprint does not match are kept on disk but not indexed
    // corrupt entries (bad magic/version, truncated) are deleted
    bool init(const server_prompt_cache_ssd_fingerprint & fp);

    // write a state to SSD atomically. returns true on success
    bool save(const server_prompt_cache_state & state);

    // find the best candidate by common prefix length, using only the in-RAM token index
    // returns nullptr when no entry qualifies, sets lcp_out to the best common prefix length
    const entry * find_best(const server_tokens & tokens_new, int & lcp_out);

    // load the state of the given entry into the slot (restores main + draft KV)
    // returns false when the restore failed
    bool load(const entry & e, server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot);

    // enforce the size limit by evicting the least recently used entries
    void update();

    size_t size() const { return total_size; }
    size_t n_entries() const { return lru.size(); }

    // cumulative event counters, copied into the server metrics
    uint64_t n_hits() const      { return n_hits_; }
    uint64_t n_misses() const    { return n_misses_; }
    uint64_t n_evictions() const { return n_evictions_; }

private:
    std::string dir;
    size_t limit_size = 0;

    server_prompt_cache_ssd_fingerprint fp;

    std::list<entry> lru;  // front = least recently used
    std::unordered_map<std::string, std::list<entry>::iterator> index;

    size_t total_size = 0;

    uint64_t n_hits_      = 0;
    uint64_t n_misses_    = 0;
    uint64_t n_evictions_ = 0;

    static std::string make_key(const server_tokens & tokens);

    bool save_file(const std::string & path, const server_prompt_cache_state & state, int64_t last_access_ms) const;
    bool read_header(const std::string & path, entry & e, server_prompt_cache_ssd_fingerprint & fp_out) const;
    bool read_state(const std::string & path, server_prompt & prompt, server_prompt_data & data) const;
    void touch_file(const std::string & path, int64_t last_access_ms) const;
};
