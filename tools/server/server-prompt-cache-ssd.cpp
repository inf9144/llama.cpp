#include "server-prompt-cache-ssd.h"

#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace {
constexpr uint32_t SSD_MAGIC   = 0x5343504c;  // "LPCS"
constexpr uint32_t SSD_VERSION = 1;
constexpr const char * SSD_EXT = ".lsc";
constexpr const char * SSD_TMP_EXT = ".lsc.tmp";
} // namespace

server_prompt_cache_ssd::server_prompt_cache_ssd(std::string dir, size_t limit_size)
    : dir(std::move(dir)), limit_size(limit_size) {
}

server_prompt_cache_ssd::~server_prompt_cache_ssd() = default;

std::string server_prompt_cache_ssd::make_key(const server_tokens & tokens) {
    const std::vector<char> packed = tokens.serialize();
    uint64_t hash = 14695981039346656037ULL;  // FNV-1a offset basis
    for (char c : packed) {
        hash ^= static_cast<uint8_t>(c);
        hash *= 1099511628211ULL;  // FNV-1a prime
    }
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(hash));
    return std::string(buf);
}

uint64_t server_prompt_cache_ssd::fp_hash() const {
    uint64_t hash = 14695981039346656037ULL;  // FNV-1a offset basis
    // hash the fields individually, the struct has padding that must not affect the hash
    auto mix = [&](const void * data, size_t size) {
        const auto * bytes = reinterpret_cast<const uint8_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            hash ^= bytes[i];
            hash *= 1099511628211ULL;  // FNV-1a prime
        }
    };
    mix(&fp.n_layer,         sizeof(fp.n_layer));
    mix(&fp.n_embd,          sizeof(fp.n_embd));
    mix(&fp.n_head,          sizeof(fp.n_head));
    mix(&fp.n_head_kv,       sizeof(fp.n_head_kv));
    mix(&fp.n_ctx_train,     sizeof(fp.n_ctx_train));
    mix(&fp.n_swa,           sizeof(fp.n_swa));
    mix(&fp.n_layer_nextn,   sizeof(fp.n_layer_nextn));
    mix(&fp.ftype,           sizeof(fp.ftype));
    mix(&fp.n_params,        sizeof(fp.n_params));
    mix(&fp.model_size,      sizeof(fp.model_size));
    mix(&fp.file_size,       sizeof(fp.file_size));
    mix(&fp.file_mtime,      sizeof(fp.file_mtime));
    mix(&fp.dft_present,     sizeof(fp.dft_present));
    mix(&fp.dft_n_layer,     sizeof(fp.dft_n_layer));
    mix(&fp.dft_n_embd,      sizeof(fp.dft_n_embd));
    mix(&fp.dft_n_head,      sizeof(fp.dft_n_head));
    mix(&fp.dft_n_head_kv,   sizeof(fp.dft_n_head_kv));
    mix(&fp.dft_ftype,       sizeof(fp.dft_ftype));
    mix(&fp.dft_n_params,    sizeof(fp.dft_n_params));
    mix(&fp.dft_model_size,  sizeof(fp.dft_model_size));
    mix(&fp.dft_file_size,   sizeof(fp.dft_file_size));
    mix(&fp.dft_file_mtime,  sizeof(fp.dft_file_mtime));
    mix(&fp.n_ctx,           sizeof(fp.n_ctx));
    mix(&fp.flash_attn,      sizeof(fp.flash_attn));
    mix(&fp.kv_unified,      sizeof(fp.kv_unified));
    mix(&fp.rope_type,       sizeof(fp.rope_type));
    mix(&fp.cache_type_k,    sizeof(fp.cache_type_k));
    mix(&fp.cache_type_v,    sizeof(fp.cache_type_v));
    mix(&fp.rope_freq_scale, sizeof(fp.rope_freq_scale));
    return hash;
}

std::string server_prompt_cache_ssd::model_dir() const {
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(fp_hash()));
    return dir + "/" + buf;
}

std::string server_prompt_cache_ssd::entry_path(const std::string & key) const {
    return model_dir() + "/" + key + SSD_EXT;
}

bool server_prompt_cache_ssd::init(const server_prompt_cache_ssd_fingerprint & fp_in) {
    fp = fp_in;

    // snapshots are stored per model, so switching models never collides
    const std::string mdir = model_dir();

    std::error_code ec;
    std::filesystem::create_directories(mdir, ec);

    for (const auto & p : std::filesystem::directory_iterator(mdir, ec)) {
        if (!p.is_regular_file()) {
            continue;
        }
        const std::string name = p.path().filename().string();

        // remove temp files left by a crashed write, only our own format
        if (string_ends_with(name, SSD_TMP_EXT)) {
            std::remove(p.path().c_str());
            continue;
        }
        if (!string_ends_with(name, SSD_EXT)) {
            continue;
        }

        entry e;
        server_prompt_cache_ssd_fingerprint fp_file;
        if (!read_header(p.path().string(), e, fp_file)) {
            SRV_WRN("ssd cache: removing corrupt entry '%s'\n", name.c_str());
            std::remove(p.path().c_str());
            continue;
        }

        // entries for a different model are kept on disk, just not indexed
        if (fp_file != fp) {
            continue;
        }

        lru.push_back(std::move(e));
        index[lru.back().key] = std::prev(lru.end());
        total_size += lru.back().size;
    }

    // order by last access, oldest first
    lru.sort([](const entry & a, const entry & b) {
        return a.last_access_ms < b.last_access_ms;
    });
    index.clear();
    for (auto it = lru.begin(); it != lru.end(); ++it) {
        index[it->key] = it;
    }

    const std::string limit_str = limit_size > 0 ? std::to_string(limit_size / (1024 * 1024)) + " MiB" : std::string("none");
    SRV_INF("ssd cache: %zu entries, %.3f MiB (limit: %s) in '%s'\n",
            lru.size(), total_size / (1024.0 * 1024.0), limit_str.c_str(), mdir.c_str());

    return true;
}

bool server_prompt_cache_ssd::save_file(const std::string & path, const server_prompt_cache_state & state, int64_t last_access_ms) const {
    const std::string tmp_path = path + ".tmp";

    std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
    if (!f) {
        SRV_ERR("ssd cache: cannot open '%s' for writing\n", tmp_path.c_str());
        return false;
    }

    auto write = [&](const void * data, size_t size) -> bool {
        f.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(size));
        return f.good();
    };

    const uint32_t magic   = SSD_MAGIC;
    const uint32_t version = SSD_VERSION;
    bool ok = write(&magic, sizeof(magic))
           && write(&version, sizeof(version))
           && write(&last_access_ms, sizeof(last_access_ms))
           && write(&fp, sizeof(fp));

    if (ok) {
        const uint8_t has_mtmd = state.prompt.tokens.has_mtmd ? 1 : 0;
        ok = write(&has_mtmd, sizeof(has_mtmd));
    }

    if (ok) {
        const std::vector<char> packed = state.prompt.tokens.serialize();
        const uint32_t n_tokens = static_cast<uint32_t>(packed.size() / sizeof(llama_token));
        ok = write(&n_tokens, sizeof(n_tokens))
           && write(packed.data(), packed.size());
    }

    if (ok) {
        const uint32_t n_ckpt = static_cast<uint32_t>(state.prompt.checkpoints.size());
        ok = write(&n_ckpt, sizeof(n_ckpt));
        for (const auto & ckpt : state.prompt.checkpoints) {
            uint64_t sz;
            ok = write(&ckpt.n_tokens, sizeof(ckpt.n_tokens))
               && write(&ckpt.id_task,  sizeof(ckpt.id_task))
               && write(&ckpt.pos_min,  sizeof(ckpt.pos_min))
               && write(&ckpt.pos_max,  sizeof(ckpt.pos_max));
            sz = ckpt.data_tgt.size();
            ok = ok && write(&sz, sizeof(sz)) && (sz == 0 || write(ckpt.data_tgt.data(), sz));
            sz = ckpt.data_dft.size();
            ok = ok && write(&sz, sizeof(sz)) && (sz == 0 || write(ckpt.data_dft.data(), sz));
            sz = ckpt.data_spec.size();
            ok = ok && write(&sz, sizeof(sz)) && (sz == 0 || write(ckpt.data_spec.data(), sz));
            if (!ok) {
                break;
            }
        }
    }

    if (ok) {
        uint64_t sz = state.data.main.size();
        ok = write(&sz, sizeof(sz)) && (sz == 0 || write(state.data.main.data(), sz));
    }
    if (ok) {
        uint64_t sz = state.data.drft.size();
        ok = write(&sz, sizeof(sz)) && (sz == 0 || write(state.data.drft.data(), sz));
    }

    if (ok) {
        f.flush();
        const uint64_t total_size = static_cast<uint64_t>(f.tellp());
        ok = write(&total_size, sizeof(total_size));
    }

    if (ok) {
        f.flush();
        ok = f.good();
    }
    f.close();

    if (!ok) {
        std::remove(tmp_path.c_str());
        SRV_ERR("ssd cache: failed to write '%s'\n", tmp_path.c_str());
        return false;
    }

    // publish atomically, the previous snapshot stays until this succeeds
    if (std::rename(tmp_path.c_str(), path.c_str()) != 0) {
        std::remove(tmp_path.c_str());
        SRV_ERR("ssd cache: failed to publish '%s'\n", path.c_str());
        return false;
    }

    return true;
}

bool server_prompt_cache_ssd::read_header(const std::string & path, entry & e, server_prompt_cache_ssd_fingerprint & fp_out) const {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }

    // total file size, used to validate the sizes read from the file
    f.seekg(0, std::ios::end);
    const uint64_t file_size = static_cast<uint64_t>(f.tellg());
    f.seekg(0, std::ios::beg);

    auto read = [&](void * data, size_t size) -> bool {
        f.read(reinterpret_cast<char *>(data), static_cast<std::streamsize>(size));
        return f.good();
    };

    uint32_t magic, version;
    if (!read(&magic, sizeof(magic)) || magic != SSD_MAGIC) {
        return false;
    }
    if (!read(&version, sizeof(version)) || version != SSD_VERSION) {
        return false;
    }

    int64_t last_access_ms;
    if (!read(&last_access_ms, sizeof(last_access_ms))) {
        return false;
    }

    server_prompt_cache_ssd_fingerprint fp_file;
    if (!read(&fp_file, sizeof(fp_file))) {
        return false;
    }

    uint8_t has_mtmd;
    if (!read(&has_mtmd, sizeof(has_mtmd))) {
        return false;
    }

    uint32_t n_tokens;
    if (!read(&n_tokens, sizeof(n_tokens))) {
        return false;
    }

    // the token data must fit in the remaining file
    const uint64_t tokens_size = static_cast<uint64_t>(n_tokens) * sizeof(llama_token);
    const uint64_t pos = static_cast<uint64_t>(f.tellg());
    if (tokens_size > file_size - pos) {
        return false;
    }

    llama_tokens packed(n_tokens);
    if (!read(packed.data(), packed.size() * sizeof(llama_token))) {
        return false;
    }

    server_tokens tokens;
    try {
        tokens = server_tokens::deserialize(packed, has_mtmd != 0);
    } catch (const std::exception &) {
        return false;
    }

    // validate the trailing total size to catch a truncated body
    if (file_size < sizeof(uint64_t)) {
        return false;
    }
    f.seekg(static_cast<std::streamoff>(file_size - sizeof(uint64_t)));
    uint64_t total_size;
    if (!read(&total_size, sizeof(total_size))) {
        return false;
    }
    if (total_size + sizeof(total_size) != file_size) {
        return false;
    }

    e.key            = make_key(tokens);
    e.size           = file_size;
    e.last_access_ms = last_access_ms;
    e.n_tokens       = tokens.size();
    e.tokens         = std::move(tokens);
    fp_out           = fp_file;

    return true;
}

bool server_prompt_cache_ssd::read_state(const std::string & path, server_prompt & prompt, server_prompt_data & data) const {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }

    f.seekg(0, std::ios::end);
    const uint64_t file_size = static_cast<uint64_t>(f.tellg());
    f.seekg(0, std::ios::beg);

    auto read = [&](void * dst, size_t size) -> bool {
        f.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(size));
        return f.good();
    };

    // check that a blob of the given size fits in the remaining file
    auto fits = [&](uint64_t size) -> bool {
        const uint64_t pos = static_cast<uint64_t>(f.tellg());
        return size <= file_size - pos;
    };

    uint32_t magic, version;
    if (!read(&magic, sizeof(magic)) || magic != SSD_MAGIC) {
        return false;
    }
    if (!read(&version, sizeof(version)) || version != SSD_VERSION) {
        return false;
    }

    int64_t last_access_ms;
    if (!read(&last_access_ms, sizeof(last_access_ms))) {
        return false;
    }

    server_prompt_cache_ssd_fingerprint fp_file;
    if (!read(&fp_file, sizeof(fp_file)) || fp_file != fp) {
        return false;
    }

    uint8_t has_mtmd;
    if (!read(&has_mtmd, sizeof(has_mtmd))) {
        return false;
    }

    uint32_t n_tokens;
    if (!read(&n_tokens, sizeof(n_tokens))) {
        return false;
    }

    const uint64_t tokens_size = static_cast<uint64_t>(n_tokens) * sizeof(llama_token);
    if (!fits(tokens_size)) {
        return false;
    }

    llama_tokens packed(n_tokens);
    if (!read(packed.data(), packed.size() * sizeof(llama_token))) {
        return false;
    }

    try {
        prompt.tokens = server_tokens::deserialize(packed, has_mtmd != 0);
    } catch (const std::exception &) {
        return false;
    }

    uint32_t n_ckpt;
    if (!read(&n_ckpt, sizeof(n_ckpt))) {
        return false;
    }
    prompt.checkpoints.clear();
    for (uint32_t i = 0; i < n_ckpt; ++i) {
        common_prompt_checkpoint ckpt;
        uint64_t sz;
        if (!read(&ckpt.n_tokens, sizeof(ckpt.n_tokens))) return false;
        if (!read(&ckpt.id_task,  sizeof(ckpt.id_task)))  return false;
        if (!read(&ckpt.pos_min,  sizeof(ckpt.pos_min)))   return false;
        if (!read(&ckpt.pos_max,  sizeof(ckpt.pos_max)))   return false;
        if (!read(&sz, sizeof(sz))) return false;
        if (!fits(sz)) return false;
        ckpt.data_tgt.resize(sz);
        if (sz && !read(ckpt.data_tgt.data(), sz)) return false;
        if (!read(&sz, sizeof(sz))) return false;
        if (!fits(sz)) return false;
        ckpt.data_dft.resize(sz);
        if (sz && !read(ckpt.data_dft.data(), sz)) return false;
        if (!read(&sz, sizeof(sz))) return false;
        if (!fits(sz)) return false;
        ckpt.data_spec.resize(sz);
        if (sz && !read(ckpt.data_spec.data(), sz)) return false;
        prompt.checkpoints.push_back(std::move(ckpt));
    }

    uint64_t main_size;
    if (!read(&main_size, sizeof(main_size))) {
        return false;
    }
    if (!fits(main_size)) {
        return false;
    }
    data.main.resize(main_size);
    if (main_size && !read(data.main.data(), main_size)) {
        return false;
    }

    uint64_t drft_size;
    if (!read(&drft_size, sizeof(drft_size))) {
        return false;
    }
    if (!fits(drft_size)) {
        return false;
    }
    data.drft.resize(drft_size);
    if (drft_size && !read(data.drft.data(), drft_size)) {
        return false;
    }

    // the trailing total size guards against truncated files
    uint64_t total_size;
    if (!read(&total_size, sizeof(total_size))) {
        return false;
    }
    if (static_cast<uint64_t>(f.tellg()) != total_size + sizeof(total_size)) {
        return false;
    }

    return true;
}

bool server_prompt_cache_ssd::save(const server_prompt_cache_state & state) {
    if (state.prompt.tokens.size() == 0) {
        return false;
    }

    const std::string key  = make_key(state.prompt.tokens);
    const std::string path = entry_path(key);
    const int64_t now = ggml_time_ms();

    if (!save_file(path, state, now)) {
        return false;
    }

    const size_t file_size = std::filesystem::file_size(path);

    auto it = index.find(key);
    if (it != index.end()) {
        total_size -= it->second->size;
        it->second->size           = file_size;
        it->second->last_access_ms = now;
        it->second->n_tokens       = state.prompt.tokens.size();
        it->second->tokens         = state.prompt.tokens.clone();
        lru.splice(lru.end(), lru, it->second);
    } else {
        entry e;
        e.key            = key;
        e.size           = file_size;
        e.last_access_ms = now;
        e.n_tokens       = state.prompt.tokens.size();
        e.tokens         = state.prompt.tokens.clone();
        lru.push_back(std::move(e));
        index[lru.back().key] = std::prev(lru.end());
    }
    total_size += file_size;

    update();

    return true;
}

const server_prompt_cache_ssd::entry * server_prompt_cache_ssd::find_best(const server_tokens & tokens_new, int & lcp_out) {
    lcp_out = -1;
    const entry * best = nullptr;

    for (const auto & e : lru) {
        const int lcp = static_cast<int>(e.tokens.get_common_prefix(tokens_new));
        const float f_keep = static_cast<float>(lcp) / static_cast<float>(e.tokens.size());
        if (f_keep < 0.25f) {
            continue;
        }
        if (lcp > lcp_out) {
            lcp_out = lcp;
            best = &e;
        }
    }

    if (best == nullptr) {
        n_misses_++;
    }

    return best;
}

bool server_prompt_cache_ssd::load(const entry & e, server_prompt & prompt, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    const std::string path = entry_path(e.key);

    server_prompt prompt_loaded;
    server_prompt_data data;
    if (!read_state(path, prompt_loaded, data)) {
        SRV_ERR("ssd cache: failed to read state '%s'\n", e.key.c_str());
        // remove the corrupt entry so it is not offered again
        std::remove(path.c_str());
        auto it = index.find(e.key);
        if (it != index.end()) {
            total_size -= it->second->size;
            lru.erase(it->second);
            index.erase(it);
        }
        return false;
    }

    {
        const size_t size = data.main.size();
        const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.main.data(), size, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (n != size) {
            SRV_ERR("ssd cache: failed to restore main state, size = %zu\n", size);
            return false;
        }
    }

    if (!data.drft.empty()) {
        GGML_ASSERT(ctx_dft);
        const size_t size = data.drft.size();
        const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.drft.data(), size, id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (n != size) {
            SRV_WRN("ssd cache: failed to restore draft state, size = %zu\n", size);
            return false;
        }
    }

    prompt = std::move(prompt_loaded);

    // mark as most recently used
    const int64_t now = ggml_time_ms();
    touch_file(path, now);
    auto it = index.find(e.key);
    if (it != index.end()) {
        it->second->last_access_ms = now;
        lru.splice(lru.end(), lru, it->second);
    }

    n_hits_++;

    return true;
}

void server_prompt_cache_ssd::update() {
    if (limit_size == 0) {
        return;
    }

    while (!lru.empty() && total_size > limit_size) {
        const std::string key = lru.front().key;
        const size_t size     = lru.front().size;
        SRV_WRN("ssd cache: size limit reached, evicting '%s' (%.3f MiB)\n", key.c_str(), size / (1024.0 * 1024.0));

        std::remove(entry_path(key).c_str());
        index.erase(key);
        total_size -= size;
        lru.pop_front();

        n_evictions_++;
    }
}

void server_prompt_cache_ssd::touch_file(const std::string & path, int64_t last_access_ms) const {
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!f) {
        return;
    }
    // last_access_ms sits right after magic(4) + version(4)
    f.seekp(sizeof(uint32_t) * 2);
    f.write(reinterpret_cast<const char *>(&last_access_ms), sizeof(last_access_ms));
    f.flush();
}
