#include "server-prompt-cache-ssd.h"

#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"
#include "gguf.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

static int n_pass = 0;
static int n_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++n_pass; } \
    else { ++n_fail; LOG_ERR("FAIL: %s (line %d)\n", msg, __LINE__); } \
} while (0)

static std::string make_temp_dir() {
    const std::string dir = "/tmp/llama-ssd-test-" + std::to_string(getpid());
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

static server_prompt_cache_ssd_fingerprint make_fp() {
    server_prompt_cache_ssd_fingerprint fp;
    fp.n_layer       = 32;
    fp.n_embd        = 4096;
    fp.n_head        = 32;
    fp.n_head_kv     = 8;
    fp.n_ctx_train   = 32768;
    fp.n_swa         = 0;
    fp.n_layer_nextn = 1;
    fp.ftype         = 2;
    fp.n_params      = 1000000000;
    fp.model_size    = 2000000000;
    fp.file_size     = 2000000000;
    fp.file_mtime    = 1700000000;
    fp.n_ctx         = 4096;
    fp.flash_attn    = 0;
    fp.kv_unified    = 0;
    return fp;
}

static server_prompt_cache_state make_state(const std::vector<llama_token> & toks, size_t main_size, size_t drft_size) {
    server_prompt_cache_state state;
    state.prompt.tokens = server_tokens(llama_tokens(toks), false);
    state.data.main.resize(main_size);
    for (size_t i = 0; i < main_size; ++i) {
        state.data.main[i] = static_cast<uint8_t>(i & 0xff);
    }
    state.data.drft.resize(drft_size);
    for (size_t i = 0; i < drft_size; ++i) {
        state.data.drft[i] = static_cast<uint8_t>((i * 7) & 0xff);
    }
    return state;
}

static common_prompt_checkpoint make_checkpoint() {
    common_prompt_checkpoint ckpt;
    ckpt.n_tokens = 3;
    ckpt.id_task = 7;
    ckpt.pos_min = 0;
    ckpt.pos_max = 2;
    ckpt.data_tgt = {1, 2, 3};
    ckpt.data_dft = {4, 5};
    ckpt.data_spec = {6, 7, 8, 9};
    return ckpt;
}

static std::vector<char> read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    CHECK(f.good(), "open snapshot file");
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static void check_stored_state(server_prompt_cache_ssd & cache, const server_prompt_cache_state & expected) {
    int lcp = -1;
    const auto * e = cache.find_best(expected.prompt.tokens, lcp);
    CHECK(e != nullptr, "stored entry discoverable");
    CHECK(lcp == static_cast<int>(expected.prompt.tokens.size()), "stored tokens match");
    if (!e) {
        return;
    }

    server_prompt prompt;
    server_prompt_data data;
    const bool ok = cache.read_state(*e, prompt, data);
    CHECK(ok, "read serialized state without a llama context");
    if (!ok) {
        return;
    }
    CHECK(prompt.tokens.serialize() == expected.prompt.tokens.serialize(), "serialized tokens intact");
    CHECK(data.main == expected.data.main, "main state bytes intact");
    CHECK(data.drft == expected.data.drft, "draft state bytes intact");
    CHECK(prompt.checkpoints.size() == expected.prompt.checkpoints.size(), "checkpoint count intact");
    auto actual = prompt.checkpoints.begin();
    for (const auto & ckpt : expected.prompt.checkpoints) {
        if (actual == prompt.checkpoints.end()) {
            break;
        }
        CHECK(actual->n_tokens == ckpt.n_tokens && actual->id_task == ckpt.id_task &&
              actual->pos_min == ckpt.pos_min && actual->pos_max == ckpt.pos_max, "checkpoint metadata intact");
        CHECK(actual->data_tgt == ckpt.data_tgt && actual->data_dft == ckpt.data_dft &&
              actual->data_spec == ckpt.data_spec, "checkpoint bytes intact");
        ++actual;
    }
}

// test the file format round-trip
static void test_roundtrip(const std::string & dir) {
    const auto fp = make_fp();
    server_prompt_cache_ssd cache(dir, 0);
    CHECK(cache.init(fp), "init");

    const std::vector<llama_token> toks = {1, 2, 3, 4, 5, 6, 7, 8};
    auto state = make_state(toks, 100, 50);
    state.prompt.checkpoints.push_back(make_checkpoint());

    CHECK(cache.save(state), "save");
    CHECK(cache.n_entries() == 1, "n_entries == 1");
    CHECK(cache.size() > 0, "size > 0");

    // verify the file exists in the model subdirectory
    const std::string mdir = cache.model_dir();
    const auto files = std::filesystem::directory_iterator(mdir);
    int n_files = 0;
    for (const auto & p : files) {
        if (p.is_regular_file() && string_ends_with(p.path().filename().string(), ".lsc")) {
            n_files++;
        }
    }
    CHECK(n_files == 1, "one .lsc file");

    // find the entry and verify the tokens
    int lcp = -1;
    const auto * e = cache.find_best(server_tokens(llama_tokens(toks), false), lcp);
    CHECK(e != nullptr, "find_best finds the entry");
    CHECK(lcp == 8, "lcp == 8");

    // use a fresh cache to read the state (simulates a restart)
    server_prompt_cache_ssd cache2(dir, 0);
    CHECK(cache2.init(fp), "init2");
    CHECK(cache2.n_entries() == 1, "n_entries2 == 1");
    check_stored_state(cache2, state);
}

static void test_save_limit_and_replacement(const std::string & dir) {
    const auto fp = make_fp();
    auto state = make_state({1, 2, 3, 4}, 100, 50);
    state.prompt.checkpoints.push_back(make_checkpoint());

    size_t snapshot_size = 0;
    std::string path;
    {
        server_prompt_cache_ssd probe(dir, 0);
        CHECK(probe.init(fp), "probe init");
        CHECK(probe.save(state), "probe save");
        int lcp = -1;
        const auto * e = probe.find_best(state.prompt.tokens, lcp);
        CHECK(e != nullptr, "probe entry");
        if (!e) {
            return;
        }
        snapshot_size = e->size;
        path = probe.model_dir() + "/" + e->key + ".lsc";
    }

    // Two snapshots fit exactly, including tokens, checkpoints and format metadata.
    const size_t limit = 2*snapshot_size;
    server_prompt_cache_ssd cache(dir, limit);
    CHECK(cache.init(fp), "limited cache init");
    auto second = make_state({9, 9, 9, 9}, 100, 50);
    second.prompt.checkpoints.push_back(make_checkpoint());
    CHECK(cache.save(second), "save at exact capacity");
    CHECK(cache.n_entries() == 2 && cache.size() == limit, "exact capacity accounting");
    CHECK(cache.n_evictions() == 0, "no eviction at capacity");
    const auto original_bytes = read_file(path);

    auto oversized = make_state({1, 2, 3, 4}, limit - 50 - make_checkpoint().size(), 50);
    oversized.prompt.checkpoints.push_back(make_checkpoint());
    CHECK(oversized.size() == limit, "payload fits but serialized metadata exceeds limit");
    const std::string tmp_path = path + ".tmp";
    {
        std::ofstream f(tmp_path, std::ios::binary);
        f << "untouched";
    }
    const auto tmp_bytes = read_file(tmp_path);
    CHECK(!cache.save(oversized), "oversized replacement rejected");
    CHECK(read_file(tmp_path) == tmp_bytes, "oversized save does not open the temp file");
    CHECK(read_file(path) == original_bytes, "oversized replacement preserves old snapshot");
    std::filesystem::remove(tmp_path);

    oversized.prompt.tokens = server_tokens(llama_tokens({8, 8, 8, 8}), false);
    CHECK(!cache.save(oversized), "oversized new snapshot rejected");
    CHECK(cache.n_entries() == 2 && cache.size() == limit && cache.n_evictions() == 0, "rejection preserves cache accounting");

    // A directory at the temp path causes a deterministic write failure, including as root.
    std::filesystem::create_directory(tmp_path);
    state.data.main[0] ^= 0xff;
    CHECK(!cache.save(state), "failed replacement reports failure");
    CHECK(read_file(path) == original_bytes, "failed write preserves old snapshot");
    CHECK(cache.n_entries() == 2 && cache.size() == limit && cache.n_evictions() == 0, "failed write preserves accounting");
    std::filesystem::remove(tmp_path);

    CHECK(cache.save(state), "normal replacement succeeds");
    CHECK(cache.n_entries() == 2 && cache.size() == limit, "replacement counted once");
    auto third = make_state({7, 7, 7, 7}, 100, 50);
    third.prompt.checkpoints.push_back(make_checkpoint());
    CHECK(cache.save(third), "normal snapshot retained after eviction");
    CHECK(cache.n_entries() == 2 && cache.size() == limit && cache.n_evictions() == 1, "LRU eviction accounting");
    int lcp = -1;
    CHECK(cache.find_best(second.prompt.tokens, lcp) == nullptr, "replacement becomes most recently used");
    check_stored_state(cache, state);
    check_stored_state(cache, third);

    server_prompt_cache_ssd restarted(dir, limit);
    CHECK(restarted.init(fp), "limited cache restart");
    CHECK(restarted.n_entries() == 2 && restarted.size() == limit, "retained snapshots indexed after restart");
    check_stored_state(restarted, state);
    check_stored_state(restarted, third);
}

// test LRU eviction
static void test_lru(const std::string & dir) {
    const auto fp = make_fp();
    // small limit to force eviction
    server_prompt_cache_ssd cache(dir, 2048);
    CHECK(cache.init(fp), "init");

    // save 3 states, each ~1000 bytes, so the limit (2048) is exceeded
    for (int i = 0; i < 3; ++i) {
        std::vector<llama_token> toks(10, 100 + i);
        auto state = make_state(toks, 500, 0);
        CHECK(cache.save(state), "save");
    }

    // after eviction, there should be fewer entries
    CHECK(cache.n_entries() < 3, "eviction happened");
    CHECK(cache.size() <= 2048, "size within limit");
}

// test fingerprint mismatch (incompatible entries are kept on disk but not indexed)
static void test_fingerprint_mismatch(const std::string & dir) {
    const auto fp = make_fp();
    server_prompt_cache_ssd cache(dir, 0);
    CHECK(cache.init(fp), "init");

    const std::vector<llama_token> toks = {1, 2, 3, 4, 5};
    auto state = make_state(toks, 100, 0);
    CHECK(cache.save(state), "save");
    CHECK(cache.n_entries() == 1, "n_entries == 1");

    // a different model (different fingerprint) should not see the entry
    auto fp2 = fp;
    fp2.n_layer = 64;  // different model
    server_prompt_cache_ssd cache2(dir, 0);
    CHECK(cache2.init(fp2), "init2");
    CHECK(cache2.n_entries() == 0, "incompatible entry not indexed");

    // the file should still be on disk (kept for when we switch back)
    int n_files = 0;
    for (const auto & p : std::filesystem::recursive_directory_iterator(dir)) {
        if (p.is_regular_file() && string_ends_with(p.path().filename().string(), ".lsc")) {
            n_files++;
        }
    }
    CHECK(n_files == 1, "file kept on disk");

    // switching back to the original model should see the entry again
    server_prompt_cache_ssd cache3(dir, 0);
    CHECK(cache3.init(fp), "init3");
    CHECK(cache3.n_entries() == 1, "entry visible again");
}

// test that LoRA and swa_full changes produce a different fingerprint
static void test_fingerprint_lora_swa(const std::string & dir) {
    const auto fp = make_fp();
    server_prompt_cache_ssd cache(dir, 0);
    CHECK(cache.init(fp), "init");

    const std::vector<llama_token> toks = {1, 2, 3, 4, 5};
    auto state = make_state(toks, 100, 0);
    CHECK(cache.save(state), "save");
    CHECK(cache.n_entries() == 1, "n_entries == 1");

    // a different LoRA adapter set should not see the entry
    auto fp_lora = fp;
    fp_lora.lora_hash = 0xdeadbeefcafe1234ULL;
    server_prompt_cache_ssd cache_lora(dir, 0);
    CHECK(cache_lora.init(fp_lora), "init_lora");
    CHECK(cache_lora.n_entries() == 0, "lora mismatch not indexed");

    // a different swa_full setting should not see the entry
    auto fp_swa = fp;
    fp_swa.swa_full = 1;
    server_prompt_cache_ssd cache_swa(dir, 0);
    CHECK(cache_swa.init(fp_swa), "init_swa");
    CHECK(cache_swa.n_entries() == 0, "swa_full mismatch not indexed");

    // switching back to the original fingerprint should see the entry again
    server_prompt_cache_ssd cache2(dir, 0);
    CHECK(cache2.init(fp), "init2");
    CHECK(cache2.n_entries() == 1, "entry visible again");
}

// test corrupt file handling
static void test_corrupt(const std::string & dir) {
    const auto fp = make_fp();
    server_prompt_cache_ssd cache(dir, 0);
    CHECK(cache.init(fp), "init");

    const std::string mdir = cache.model_dir();

    // create a corrupt file (bad magic)
    const std::string corrupt_path = mdir + "/deadbeefdeadbeefdeadbeefdeadbeef.lsc";
    {
        std::ofstream f(corrupt_path, std::ios::binary);
        f.write("garbage", 7);
    }

    // re-init, the corrupt file should be removed
    server_prompt_cache_ssd cache2(dir, 0);
    CHECK(cache2.init(fp), "init2");
    CHECK(!std::filesystem::exists(corrupt_path), "corrupt file removed");

    // create a truncated file (valid header, truncated body)
    const std::vector<llama_token> toks = {1, 2, 3, 4, 5};
    auto state = make_state(toks, 100, 0);
    CHECK(cache2.save(state), "save");

    // find the .lsc file
    std::string valid_path;
    for (const auto & p : std::filesystem::directory_iterator(mdir)) {
        if (p.is_regular_file() && string_ends_with(p.path().filename().string(), ".lsc")) {
            valid_path = p.path().string();
            break;
        }
    }
    CHECK(!valid_path.empty(), "found .lsc file");

    // read the full content first, then write only half (truncating the body)
    {
        std::ifstream in(valid_path, std::ios::binary);
        std::vector<char> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        in.close();
        std::ofstream f(valid_path, std::ios::binary | std::ios::trunc);
        f.write(data.data(), data.size() / 2);  // write only half
    }

    // re-init, the truncated file should be removed
    server_prompt_cache_ssd cache3(dir, 0);
    CHECK(cache3.init(fp), "init3");
    CHECK(!std::filesystem::exists(valid_path), "truncated file removed");
}

// test that snapshots written by an older format version are invalidated, not reinterpreted
static void test_version_mismatch(const std::string & dir) {
    const auto fp = make_fp();
    server_prompt_cache_ssd cache(dir, 0);
    CHECK(cache.init(fp), "init");

    auto old_state = make_state({1, 2, 3, 4, 5}, 100, 50);
    old_state.prompt.checkpoints.push_back(make_checkpoint());
    auto current_state = make_state({9, 8, 7, 6, 5}, 200, 75);
    CHECK(cache.save(old_state), "save snapshot to downgrade");
    CHECK(cache.save(current_state), "save current-version control");

    server_prompt_cache_ssd valid(dir, 0);
    CHECK(valid.init(fp), "index valid snapshots");
    CHECK(valid.n_entries() == 2, "both current snapshots indexed");
    check_stored_state(valid, old_state);
    check_stored_state(valid, current_state);
    int lcp = -1;
    const auto * e = valid.find_best(old_state.prompt.tokens, lcp);
    CHECK(e != nullptr, "find snapshot to downgrade");
    if (!e) {
        return;
    }
    const std::string old_path = valid.model_dir() + "/" + e->key + ".lsc";
    const auto original = read_file(old_path);
    const size_t version_offset = sizeof(uint32_t);
    uint32_t version = 0;
    if (original.size() < version_offset + sizeof(version)) {
        CHECK(false, "complete version field");
        return;
    }
    std::memcpy(&version, original.data() + version_offset, sizeof(version));
    CHECK(version == 2, "production serializer writes version 2");
    --version;
    {
        std::fstream f(old_path, std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(version_offset);
        f.write(reinterpret_cast<const char *>(&version), sizeof(version));
        CHECK(f.good(), "patch only the version field");
    }
    auto expected_bytes = original;
    std::memcpy(expected_bytes.data() + version_offset, &version, sizeof(version));
    CHECK(read_file(old_path) == expected_bytes, "remaining serialized structure unchanged");

    server_prompt prompt;
    server_prompt_data data;
    CHECK(!valid.read_state(*e, prompt, data), "state reader rejects incompatible version");

    // re-init, the old-version file must be removed, not indexed
    server_prompt_cache_ssd cache2(dir, 0);
    CHECK(cache2.init(fp), "init2");
    CHECK(!std::filesystem::exists(old_path), "old-version file removed");
    CHECK(cache2.n_entries() == 1, "only current-version snapshot indexed");
    CHECK(cache2.find_best(old_state.prompt.tokens, lcp) == nullptr, "old-version snapshot not discoverable");
    check_stored_state(cache2, current_state);
}

// Test RAM allocation and eviction separately from the direct save_state() fallback.
static void test_ram_alloc_eviction(const std::string & dir) {
    const auto fp = make_fp();
    server_prompt_cache_ssd ssd(dir, 0);
    CHECK(ssd.init(fp), "ssd init");

    // 1 MiB RAM limit
    server_prompt_cache ram(1, 0, &ssd);

    server_prompt prompt;
    prompt.tokens = server_tokens(llama_tokens({1, 2, 3, 4, 5}), false);

    // a state larger than the RAM limit: alloc() must fail
    const size_t size_too_big = 2 * 1024 * 1024;  // 2 MiB > 1 MiB limit
    CHECK(ram.alloc(prompt, size_too_big, 0) == nullptr, "alloc fails for oversized state");
    CHECK(!ram.is_in_cache(prompt), "oversized state not reported as cached");

    // a state that fits: alloc() succeeds and the prompt becomes cached
    const size_t size_half = 600 * 1024;  // 600 KiB
    auto * st = ram.alloc(prompt, size_half, 0);
    CHECK(st != nullptr, "alloc succeeds for fitting state");
    CHECK(ram.is_in_cache(prompt), "state reported as cached");

    // a second alloc for the same prompt: already cached, returns nullptr
    CHECK(ram.alloc(prompt, size_half, 0) == nullptr, "second alloc returns nullptr (already cached)");

    // a second, unrelated state: together they exceed the limit, so the first is evicted to SSD
    server_prompt prompt2;
    prompt2.tokens = server_tokens(llama_tokens({9, 8, 7, 6, 5}), false);
    auto * st2 = ram.alloc(prompt2, size_half, 0);
    CHECK(st2 != nullptr, "alloc succeeds for second state");
    CHECK(ssd.n_entries() == 1, "evicted state saved to SSD");
}

static constexpr size_t TEST_STATE_SIZE = 2*1024*1024;
static constexpr llama_seq_id TEST_SLOT_ID = 7;

// Replace only llama state capture. RAM allocation, SSD saving and reading remain real.
static void use_test_state_reader(server_prompt_cache & cache) {
    cache.get_state_size = [](llama_context *, llama_seq_id id, llama_state_seq_flags flags) -> size_t {
        CHECK(id == TEST_SLOT_ID && flags == LLAMA_STATE_SEQ_FLAGS_NONE, "state size arguments");
        return TEST_STATE_SIZE;
    };
    cache.get_state_data = [](llama_context *, uint8_t * dst, size_t size, llama_seq_id id, llama_state_seq_flags flags) -> size_t {
        CHECK(size == TEST_STATE_SIZE && id == TEST_SLOT_ID && flags == LLAMA_STATE_SEQ_FLAGS_NONE, "state read arguments");
        for (size_t i = 0; i < size; ++i) {
            dst[i] = static_cast<uint8_t>(i & 0xff);
        }
        return size;
    };
}

static void test_ram_limit_ssd_fallback(const std::string & dir) {
    const auto fp = make_fp();
    server_prompt_cache_ssd ssd(dir, 0);
    CHECK(ssd.init(fp), "ssd init");
    auto state = make_state({1, 2, 3, 4, 5}, TEST_STATE_SIZE, 0);
    state.prompt.checkpoints.push_back(make_checkpoint());

    server_prompt_cache ram(1, 0, &ssd);
    use_test_state_reader(ram);
    CHECK(state.size() > ram.limit_size, "snapshot exceeds RAM capacity");
    CHECK(ram.alloc(state.prompt, TEST_STATE_SIZE, 0) == nullptr, "RAM cannot retain snapshot");
    CHECK(ram.save_state(state.prompt, nullptr, nullptr, TEST_SLOT_ID), "save_state directly persists oversized RAM snapshot to SSD");
    CHECK(ram.states.empty() && !ram.is_in_cache(state.prompt), "snapshot not retained in RAM");
    CHECK(ssd.n_entries() == 1 && ssd.n_evictions() == 0, "direct save without RAM eviction");
    check_stored_state(ssd, state);
    server_prompt_cache_ssd restarted(dir, 0);
    CHECK(restarted.init(fp), "fallback snapshot restart");
    check_stored_state(restarted, state);

    server_prompt_cache no_ssd(1, 0);
    use_test_state_reader(no_ssd);
    CHECK(!no_ssd.save_state(state.prompt, nullptr, nullptr, TEST_SLOT_ID), "no SSD fallback reports failure");
    CHECK(no_ssd.states.empty(), "failed save leaves RAM empty");

    server_prompt_cache_ssd too_small(dir + "/small", 1024);
    CHECK(too_small.init(fp), "small SSD init");
    server_prompt_cache limited(1, 0, &too_small);
    use_test_state_reader(limited);
    CHECK(!limited.save_state(state.prompt, nullptr, nullptr, TEST_SLOT_ID), "oversized SSD fallback reports failure");
    CHECK(limited.states.empty() && too_small.n_entries() == 0 && too_small.size() == 0 && too_small.n_evictions() == 0, "oversized fallback retains nothing");

    server_prompt_cache_ssd unwritable(dir + "/blocked", 0);
    CHECK(unwritable.init(fp), "blocked SSD init");
    std::filesystem::remove(unwritable.model_dir());
    {
        std::ofstream f(unwritable.model_dir());
        f << "block snapshot directory";
    }
    server_prompt_cache failed(1, 0, &unwritable);
    use_test_state_reader(failed);
    CHECK(!failed.save_state(state.prompt, nullptr, nullptr, TEST_SLOT_ID), "failed SSD write reports failure");
    CHECK(failed.states.empty() && unwritable.n_entries() == 0 && unwritable.size() == 0, "failed fallback retains nothing");

    ram.get_state_data = [](llama_context *, uint8_t *, size_t, llama_seq_id, llama_state_seq_flags) -> size_t { return 0; };
    auto other = state.prompt.clone();
    other.tokens = server_tokens(llama_tokens({9, 9, 9, 9, 9}), false);
    CHECK(!ram.save_state(other, nullptr, nullptr, TEST_SLOT_ID), "failed state capture reports failure before SSD save");
    CHECK(ram.states.empty() && ssd.n_entries() == 1, "failed capture does not publish a snapshot");

    server_prompt_cache fitting(3, 0);
    use_test_state_reader(fitting);
    fitting.get_state_data = ram.get_state_data;
    CHECK(!fitting.save_state(state.prompt, nullptr, nullptr, TEST_SLOT_ID), "failed RAM state capture reports failure");
    CHECK(fitting.states.empty() && !fitting.is_in_cache(state.prompt), "failed RAM capture is not reusable");
    use_test_state_reader(fitting);
    CHECK(fitting.save_state(state.prompt, nullptr, nullptr, TEST_SLOT_ID), "normal RAM save_state succeeds");
    CHECK(fitting.is_in_cache(state.prompt) && fitting.states.back().data.main == state.data.main, "normal RAM state bytes intact");

    server_prompt_cache disabled(0, 0, &ssd);
    use_test_state_reader(disabled);
    CHECK(disabled.alloc(state.prompt, 1, 0) == nullptr, "zero RAM limit disables even small RAM entries");
    CHECK(disabled.save_state(state.prompt, nullptr, nullptr, TEST_SLOT_ID), "zero RAM limit still saves to SSD");
    CHECK(disabled.states.empty() && disabled.size() == 0, "SSD-only mode retains no RAM snapshots");

    server_prompt_cache no_cache(0, 0);
    use_test_state_reader(no_cache);
    CHECK(!no_cache.save_state(state.prompt, nullptr, nullptr, TEST_SLOT_ID), "both tiers disabled retain nothing");
    CHECK(no_cache.states.empty(), "both tiers disabled leave RAM empty");

    server_prompt_cache unlimited(-1, 0);
    CHECK(unlimited.alloc(state.prompt, 1, 0) != nullptr, "negative RAM limit still allows RAM entries");
}

static void test_unchanged_ssd_snapshot(const std::string & dir) {
    server_prompt_cache_ssd ssd(dir, 3*1024*1024);
    CHECK(ssd.init(make_fp()), "unchanged snapshot SSD init");
    server_prompt_cache cache(0, 0, &ssd);
    use_test_state_reader(cache);
    auto state = make_state({1, 2, 3, 4, 5}, TEST_STATE_SIZE, 0);
    state.prompt.checkpoints.push_back(make_checkpoint());
    CHECK(cache.save_state(state.prompt, nullptr, nullptr, TEST_SLOT_ID), "save initial SSD-only snapshot");
    CHECK(ssd.is_saved(state.prompt), "live prompt remembers retained snapshot");
    int lcp;
    const auto * entry = ssd.find_best(state.prompt.tokens, lcp);
    CHECK(entry != nullptr, "unchanged snapshot indexed");
    if (!entry) {
        return;
    }
    const std::string path = ssd.model_dir() + "/" + entry->key + ".lsc";
    const auto fixed_time = std::filesystem::file_time_type{};
    std::filesystem::last_write_time(path, fixed_time);
    cache.get_state_size = [](llama_context *, llama_seq_id, llama_state_seq_flags) -> size_t {
        CHECK(false, "unchanged snapshot must skip state capture");
        return 0;
    };
    CHECK(cache.save_state(state.prompt, nullptr, nullptr, TEST_SLOT_ID), "repeat save reuses unchanged SSD snapshot");
    CHECK(ssd.save_context(state.prompt, nullptr, nullptr, TEST_SLOT_ID), "context save also skips unchanged snapshot");
    CHECK(std::filesystem::last_write_time(path) == fixed_time, "repeat saves do not write the snapshot file");

    use_test_state_reader(cache);
    state.prompt.invalidate_ssd_snapshot();
    state.prompt.checkpoints.front().data_spec.push_back(42);
    CHECK(!ssd.is_saved(state.prompt), "modified prompt invalidates saved identity despite equal tokens");
    CHECK(cache.save_state(state.prompt, nullptr, nullptr, TEST_SLOT_ID), "modified same-token state is saved again");
    CHECK(std::filesystem::last_write_time(path) != fixed_time, "modified same-token state rewrites snapshot");
    check_stored_state(ssd, state);

    auto replacement = make_state({1, 2, 3, 4, 5}, TEST_STATE_SIZE, 0);
    CHECK(ssd.save(replacement), "another producer replaces same token key");
    CHECK(!ssd.is_saved(state.prompt), "replacement invalidates previous live snapshot identity");
    CHECK(cache.save_state(state.prompt, nullptr, nullptr, TEST_SLOT_ID), "re-save after same-key replacement");
    check_stored_state(ssd, state);

    auto other = make_state({9, 9, 9, 9, 9}, TEST_STATE_SIZE, 0);
    CHECK(ssd.save(other) && ssd.n_entries() == 1, "evict previously saved snapshot");
    CHECK(!ssd.is_saved(state.prompt), "evicted snapshot is no longer reusable");
    CHECK(cache.save_state(state.prompt, nullptr, nullptr, TEST_SLOT_ID), "re-save after eviction");
    CHECK(ssd.is_saved(state.prompt), "re-saved snapshot identity is valid");

    std::filesystem::remove(path);
    CHECK(!ssd.is_saved(state.prompt), "missing snapshot cannot suppress a write");
    CHECK(cache.save_state(state.prompt, nullptr, nullptr, TEST_SLOT_ID), "recreate missing snapshot");
    CHECK(ssd.is_saved(state.prompt), "recreated snapshot has a valid identity");
    server_prompt_cache_ssd reopened(dir, 0);
    CHECK(reopened.init(make_fp()), "reopen snapshot index");
    CHECK(!reopened.is_saved(state.prompt), "a different cache instance cannot match an old runtime identity");
    state.prompt.clear();
    CHECK(state.prompt.ssd_snapshot_id == 0, "clearing prompt invalidates snapshot identity");
}

// A small deterministic model exercises real KV transfers without a download.
static llama_model_ptr make_test_model(const std::string & path) {
    gguf_context * meta = gguf_init_empty();
    gguf_set_val_str(meta, "general.architecture", "llama");
    gguf_set_val_str(meta, "tokenizer.ggml.model", "none");
    gguf_set_val_u32(meta, "llama.vocab_size", 64);
    gguf_set_val_u32(meta, "llama.context_length", 256);
    gguf_set_val_u32(meta, "llama.embedding_length", 64);
    gguf_set_val_u32(meta, "llama.feed_forward_length", 128);
    gguf_set_val_u32(meta, "llama.block_count", 2);
    gguf_set_val_u32(meta, "llama.attention.head_count", 2);
    gguf_set_val_u32(meta, "llama.attention.head_count_kv", 1);
    gguf_set_val_f32(meta, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    ggml_init_params init = {2*1024*1024, nullptr, false};
    ggml_context * weights = ggml_init(init);
    int seed = 0;
    auto add_tensor = [&](const std::string & name, int64_t nx, int64_t ny = 0) {
        auto * tensor = ny ? ggml_new_tensor_2d(weights, GGML_TYPE_F32, nx, ny) : ggml_new_tensor_1d(weights, GGML_TYPE_F32, nx);
        ggml_set_name(tensor, name.c_str());
        auto * data = static_cast<float *>(tensor->data);
        for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
            data[i] = name.find("norm") != std::string::npos ? 1.0f : 0.05f*std::sin(float(i + ++seed));
        }
        gguf_add_tensor(meta, tensor);
    };
    add_tensor("token_embd.weight", 64, 64);
    add_tensor("output_norm.weight", 64);
    add_tensor("output.weight", 64, 64);
    for (int layer = 0; layer < 2; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        add_tensor(prefix + "attn_norm.weight", 64);
        add_tensor(prefix + "attn_q.weight", 64, 64);
        add_tensor(prefix + "attn_k.weight", 64, 32);
        add_tensor(prefix + "attn_v.weight", 64, 32);
        add_tensor(prefix + "attn_output.weight", 64, 64);
        add_tensor(prefix + "ffn_norm.weight", 64);
        add_tensor(prefix + "ffn_gate.weight", 64, 128);
        add_tensor(prefix + "ffn_down.weight", 128, 64);
        add_tensor(prefix + "ffn_up.weight", 64, 128);
    }
    CHECK(gguf_write_to_file(meta, path.c_str(), false), "write test model");
    gguf_free(meta);
    ggml_free(weights);
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    return llama_model_ptr(llama_model_load_from_file(path.c_str(), params));
}

static std::vector<uint8_t> capture_state(llama_context * ctx, llama_seq_id seq_id = 0) {
    std::vector<uint8_t> state(llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE));
    CHECK(!state.empty(), "real state has a size");
    CHECK(llama_state_seq_get_data_ext(ctx, state.data(), state.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == state.size(), "capture real sequence");
    return state;
}

static void decode_test_tokens(llama_context * ctx, const llama_tokens & tokens, llama_pos pos = 0, llama_seq_id seq_id = 0) {
    common_batch batch(ctx);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.add(tokens[i], pos + i, seq_id, true);
    }
    CHECK(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "decode test tokens");
}

static void test_io_modes(const std::string & dir) {
    std::filesystem::create_directories(dir);
    auto model = make_test_model(dir + "/model.gguf");
    CHECK(model != nullptr, "load deterministic test model");
    if (!model) {
        return;
    }
    for (const auto kv_type : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
        auto params = llama_context_default_params();
        params.n_ctx = 256;
        params.n_batch = params.n_ubatch = 64;
        params.n_seq_max = 2;
        params.n_threads = params.n_threads_batch = 2;
        params.kv_unified = true;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        params.type_k = params.type_v = kv_type;
        auto make_ctx = [&]() { return llama_context_ptr(llama_init_from_model(model.get(), params)); };
        auto source = make_ctx();
        auto draft = make_ctx();
        CHECK(source && draft, "create real target and draft contexts");
        if (!source || !draft) {
            continue;
        }
        const llama_tokens tokens = {1, 2, 3, 4, 5, 6, 7, 8};
        decode_test_tokens(source.get(), tokens);
        decode_test_tokens(draft.get(), {8, 7, 6, 5, 4, 3, 2, 1});
        server_prompt_cache_state expected;
        expected.prompt.tokens = server_tokens(tokens, false);
        expected.prompt.checkpoints.push_back(make_checkpoint());
        expected.data.main = capture_state(source.get());
        expected.data.drft = capture_state(draft.get());

        decode_test_tokens(source.get(), {9}, 8);
        const float * logits = llama_get_logits_ith(source.get(), -1);
        std::vector<float> reference(logits, logits + 64);
        CHECK(llama_state_seq_set_data_ext(source.get(), expected.data.main.data(), expected.data.main.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE) == expected.data.main.size(), "reset reference state");

        for (const bool with_draft : {false, true}) {
            server_prompt_cache_state state;
            state.prompt = expected.prompt.clone();
            state.data = expected.data;
            if (!with_draft) {
                state.data.drft.clear();
            }
            for (const bool save_streaming : {false, true}) {
                const std::string cache_dir = dir + "/" + std::to_string(kv_type) + "-" + std::to_string(with_draft) + "-" + std::to_string(save_streaming);
                server_prompt_cache_ssd writer(cache_dir, 0, save_streaming, 128);
                CHECK(writer.init(make_fp()), "initialize real-state writer");
                CHECK(writer.save_context(state.prompt, source.get(), with_draft ? draft.get() : nullptr, 0), "save real state in selected mode");
                CHECK(writer.is_saved(state.prompt), "context save remembers exact SSD snapshot");
                check_stored_state(writer, state);
                int lcp;
                const auto * saved_entry = writer.find_best(state.prompt.tokens, lcp);
                CHECK(saved_entry != nullptr, "context snapshot indexed");
                if (saved_entry) {
                    const std::string path = writer.model_dir() + "/" + saved_entry->key + ".lsc";
                    const auto fixed_time = std::filesystem::file_time_type{};
                    std::filesystem::last_write_time(path, fixed_time);
                    CHECK(writer.save_context(state.prompt, source.get(), with_draft ? draft.get() : nullptr, 0), "repeated context save succeeds");
                    CHECK(std::filesystem::last_write_time(path) == fixed_time, "repeated context save skips SSD write in both modes");
                }
                state.prompt.invalidate_ssd_snapshot();
                server_prompt_cache ssd_only(0, 0, &writer);
                if (save_streaming) {
                    ssd_only.get_state_data = [](llama_context *, uint8_t *, size_t, llama_seq_id, llama_state_seq_flags) -> size_t {
                        CHECK(false, "streaming must not use buffered state capture");
                        return 0;
                    };
                }
                CHECK(ssd_only.save_state(state.prompt, source.get(), with_draft ? draft.get() : nullptr, 0), "SSD-only coordinator saves real state");
                CHECK(ssd_only.states.empty() && ssd_only.size() == 0, "SSD-only coordinator retains no RAM state");

                for (const bool load_streaming : {false, true}) {
                    auto target = make_ctx();
                    auto target_draft = make_ctx();
                    CHECK(target && target_draft, "create restore contexts");
                    if (!target || !target_draft) {
                        continue;
                    }
                    decode_test_tokens(target.get(), {20, 21, 22}, 0, 1);
                    const auto other_sequence = capture_state(target.get(), 1);
                    server_prompt_cache_ssd reader(cache_dir, 0, load_streaming, 192);
                    CHECK(reader.init(make_fp()), "reopen real-state cache in selected mode");
                    int lcp = -1;
                    const auto * entry = reader.find_best(state.prompt.tokens, lcp);
                    CHECK(entry != nullptr, "real state discoverable after reopening");
                    if (!entry) {
                        continue;
                    }
                    server_prompt restored;
                    CHECK(reader.load(*entry, restored, target.get(), with_draft ? target_draft.get() : nullptr, 0), "cross-mode restore succeeds");
                    CHECK(reader.n_hits() == 1, "successful restore counts as a hit");
                    CHECK(capture_state(target.get()) == state.data.main, "cross-mode target state is byte-identical");
                    CHECK(capture_state(target.get(), 1) == other_sequence, "restore preserves unrelated sequence");
                    if (with_draft) {
                        CHECK(capture_state(target_draft.get()) == state.data.drft, "cross-mode draft state is byte-identical");
                    }
                    CHECK(restored.tokens.get_common_prefix(state.prompt.tokens) == tokens.size(), "cross-mode tokens intact");
                    CHECK(restored.checkpoints.size() == 1 && restored.checkpoints.front().data_spec == state.prompt.checkpoints.front().data_spec, "cross-mode checkpoint state intact");
                    decode_test_tokens(target.get(), {9}, 8);
                    logits = llama_get_logits_ith(target.get(), -1);
                    for (size_t i = 0; i < reference.size(); ++i) {
                        CHECK(std::isfinite(logits[i]) && std::abs(logits[i] - reference[i]) < 1e-5f, "cross-mode continuation logits match");
                    }

                    // The container stays valid; each fault is inside a sequence-state payload.
                    for (int fault = 0; fault < 11; ++fault) {
                        server_prompt_cache_state corrupt;
                        corrupt.prompt = expected.prompt.clone();
                        corrupt.data = expected.data;
                        // This model has one KV stream and 12 metadata bytes per token, without cell extensions.
                        const size_t tensor_header = 16 + 12*tokens.size();
                        switch (fault) {
                            case 0: corrupt.data.main.pop_back(); break;
                            case 1: corrupt.data.drft[0] ^= 0xff; break;
                            case 2: corrupt.data.main[8] ^= 0xff; break; // stream count
                            case 3: corrupt.data.main[20] ^= 0xff; break; // cell sequence count
                            case 4: corrupt.data.main[tensor_header + 4] ^= 0xff; break; // layer count
                            case 5: corrupt.data.main[tensor_header + 20] ^= 0xff; break; // key row size
                            case 6: corrupt.data.drft.pop_back(); break;
                            case 7: corrupt.data.main.clear(); break;
                            case 8: corrupt.data.main.resize(8); break; // missing KV metadata
                            case 9: corrupt.data.main.push_back(0); break; // unconsumed payload bytes
                            case 10: corrupt.data.main[13] ^= 0xff; break; // cell count exceeds total capacity
                        }
                        CHECK(reader.save(corrupt), "publish malformed state inside valid container");
                        entry = reader.find_best(corrupt.prompt.tokens, lcp);
                        CHECK(entry != nullptr, "malformed state indexed before restore");
                        if (!entry) {
                            continue;
                        }
                        const std::string path = reader.model_dir() + "/" + entry->key + ".lsc";
                        CHECK(!reader.load(*entry, restored, target.get(), target_draft.get(), 0), "malformed state restore rejected");
                        CHECK(restored.tokens.empty() && restored.checkpoints.empty(), "failed restore clears prompt metadata");
                        CHECK(llama_memory_seq_pos_max(llama_get_memory(target.get()), 0) < 0, "failed restore clears target sequence");
                        CHECK(llama_memory_seq_pos_max(llama_get_memory(target_draft.get()), 0) < 0, "failed restore clears draft sequence");
                        CHECK(capture_state(target.get(), 1) == other_sequence, "failed restore preserves unrelated sequence");
                        CHECK(reader.n_hits() == 1 && reader.n_entries() == 0 && reader.size() == 0, "invalid payload is removed without counting a hit");
                        CHECK(!std::filesystem::exists(path), "invalid payload file is removed");
                        CHECK(reader.find_best(corrupt.prompt.tokens, lcp) == nullptr, "invalid payload is not offered again");
                    }
                    CHECK(reader.save(state), "publish valid state before truncating its container");
                    entry = reader.find_best(state.prompt.tokens, lcp);
                    CHECK(entry != nullptr, "valid state indexed before container corruption");
                    if (entry) {
                        const std::string path = reader.model_dir() + "/" + entry->key + ".lsc";
                        std::filesystem::resize_file(path, entry->size - 1);
                        CHECK(!reader.load(*entry, restored, target.get(), target_draft.get(), 0), "truncated container rejected");
                        CHECK(reader.n_entries() == 0 && !std::filesystem::exists(path), "proven invalid container is removed");
                    }
                    CHECK(writer.save(state), "restore valid snapshot for next reader mode");
                }
            }
        }

        for (const bool streaming : {false, true}) {
            const std::string cache_dir = dir + "/capacity-" + std::to_string(kv_type) + "-" + std::to_string(streaming);
            server_prompt_cache_ssd cache(cache_dir, 0, streaming, 192);
            CHECK(cache.init(make_fp()) && cache.save(expected), "save snapshot for capacity failure test");
            for (const bool full_draft : {false, true}) {
                auto target = make_ctx();
                auto target_draft = make_ctx();
                CHECK(target && target_draft, "create capacity test contexts");
                if (!target || !target_draft) {
                    continue;
                }
                auto * full = full_draft ? target_draft.get() : target.get();
                const int n_fill = llama_n_ctx(full) - tokens.size() + 1;
                for (int pos = 0; pos < n_fill; pos += 64) {
                    decode_test_tokens(full, llama_tokens(std::min(64, n_fill - pos), 20), pos, 1);
                }
                const auto unrelated = capture_state(full, 1);
                int lcp;
                const auto * entry = cache.find_best(expected.prompt.tokens, lcp);
                CHECK(entry != nullptr, "capacity test snapshot indexed");
                if (!entry) {
                    continue;
                }
                const std::string path = cache.model_dir() + "/" + entry->key + ".lsc";
                const auto before = read_file(path);
                const auto hits_before = cache.n_hits();
                server_prompt restored;
                CHECK(!cache.load(*entry, restored, target.get(), target_draft.get(), 0), "insufficient KV capacity rejects restore");
                CHECK(restored.tokens.empty() && restored.ssd_snapshot_id == 0, "capacity failure clears prompt identity");
                CHECK(llama_memory_seq_pos_max(llama_get_memory(target.get()), 0) < 0 &&
                      llama_memory_seq_pos_max(llama_get_memory(target_draft.get()), 0) < 0, "capacity failure clears both destination sequences");
                CHECK(capture_state(full, 1) == unrelated, "capacity failure preserves unrelated sequence");
                CHECK(cache.n_entries() == 1 && cache.n_hits() == hits_before, "capacity failure retains index and does not count a hit");
                CHECK(read_file(path) == before, "capacity failure leaves valid snapshot intact");

                auto corrupt = server_prompt_cache_state{expected.prompt.clone(), expected.data};
                corrupt.data.drft[0] ^= 0xff;
                CHECK(cache.save(corrupt), "publish invalid draft header with occupied KV cache");
                entry = cache.find_best(expected.prompt.tokens, lcp);
                CHECK(entry && !cache.load(*entry, restored, target.get(), target_draft.get(), 0), "invalid draft header rejected before main restore");
                CHECK(cache.n_entries() == 0 && !std::filesystem::exists(path), "invalid draft header is removed even when main KV is full");
                CHECK(capture_state(full, 1) == unrelated, "header rejection preserves unrelated sequence");
                CHECK(cache.save(expected), "publish valid snapshot for capacity retry");
                llama_memory_seq_rm(llama_get_memory(full), 1, -1, -1);
                entry = cache.find_best(expected.prompt.tokens, lcp);
                CHECK(entry && cache.load(*entry, restored, target.get(), target_draft.get(), 0), "restore succeeds after releasing KV cells");
                CHECK(capture_state(target.get()) == expected.data.main && capture_state(target_draft.get()) == expected.data.drft, "retry restores byte-identical main and draft states");
                CHECK(cache.is_saved(restored), "successful retry remembers retained snapshot");
            }
        }
    }
}

static void test_state_size_failure(const std::string & dir, bool fail_draft) {
    server_prompt_cache_ssd ssd(dir, 0);
    CHECK(ssd.init(make_fp()), "state size failure SSD init");
    server_prompt prompt;
    prompt.tokens = server_tokens(llama_tokens({1, 2, 3, 4, 5}), false);

    struct state_source {
        size_t size;
        size_t size_calls = 0;
        size_t data_calls = 0;
    };

    // These opaque handles are used only by the injected readers, never by llama.
    for (const size_t limit : {size_t(1), size_t(1024*1024)}) {
        state_source target = {fail_draft ? size_t(16) : size_t(0)};
        state_source draft  = {fail_draft ? size_t(0) : size_t(16)};
        server_prompt_cache ram(1, 0, &ssd);
        ram.limit_size = limit;
        ram.get_state_size = [](llama_context * ctx, llama_seq_id id, llama_state_seq_flags flags) -> size_t {
            CHECK(id == TEST_SLOT_ID && flags == LLAMA_STATE_SEQ_FLAGS_NONE, "failed size read arguments");
            auto & source = *reinterpret_cast<state_source *>(ctx);
            ++source.size_calls;
            return source.size;
        };
        ram.get_state_data = [](llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id, llama_state_seq_flags) -> size_t {
            auto & source = *reinterpret_cast<state_source *>(ctx);
            ++source.data_calls;
            std::fill_n(dst, size, uint8_t(1));
            return size;
        };

        CHECK(!ram.save_state(prompt, reinterpret_cast<llama_context *>(&target), reinterpret_cast<llama_context *>(&draft), TEST_SLOT_ID),
              fail_draft ? "zero draft state size rejected" : "zero target state size rejected");
        CHECK((fail_draft ? draft.size_calls : target.size_calls) == 1, "failing size reader was called");
        CHECK(target.data_calls == 0 && draft.data_calls == 0, "size failure stops before state capture");
        CHECK(ram.states.empty() && ram.size() == 0 && !ram.is_in_cache(prompt), "size failure creates no RAM entry");
        CHECK(ssd.n_entries() == 0 && ssd.size() == 0 && ssd.n_evictions() == 0, "size failure creates no SSD entry");
        CHECK(std::filesystem::is_empty(ssd.model_dir()), "size failure writes no snapshot files");
    }
}

// test temp file cleanup
static void test_tmp_cleanup(const std::string & dir) {
    const auto fp = make_fp();
    server_prompt_cache_ssd cache(dir, 0);
    CHECK(cache.init(fp), "init");

    const std::string mdir = cache.model_dir();

    // create a temp file (simulates a crashed write)
    const std::string tmp_path = mdir + "/deadbeefdeadbeefdeadbeefdeadbeef.lsc.tmp";
    {
        std::ofstream f(tmp_path, std::ios::binary);
        f.write("partial", 7);
    }

    // re-init, the tmp file should be removed
    server_prompt_cache_ssd cache2(dir, 0);
    CHECK(cache2.init(fp), "init2");
    CHECK(!std::filesystem::exists(tmp_path), "tmp file removed");
}

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    const std::string dir = make_temp_dir();

    test_roundtrip(dir);
    std::filesystem::remove_all(dir);

    const std::string dir2 = make_temp_dir() + "-lru";
    std::filesystem::create_directories(dir2);
    test_lru(dir2);
    std::filesystem::remove_all(dir2);

    const std::string dir2b = make_temp_dir() + "-save-limit";
    std::filesystem::create_directories(dir2b);
    test_save_limit_and_replacement(dir2b);
    std::filesystem::remove_all(dir2b);

    const std::string dir3 = make_temp_dir() + "-fp";
    std::filesystem::create_directories(dir3);
    test_fingerprint_mismatch(dir3);
    std::filesystem::remove_all(dir3);

    const std::string dir4 = make_temp_dir() + "-corrupt";
    std::filesystem::create_directories(dir4);
    test_corrupt(dir4);
    std::filesystem::remove_all(dir4);

    const std::string dir4c = make_temp_dir() + "-version";
    std::filesystem::create_directories(dir4c);
    test_version_mismatch(dir4c);
    std::filesystem::remove_all(dir4c);

    const std::string dir4d = make_temp_dir() + "-ram-limit";
    std::filesystem::create_directories(dir4d);
    test_ram_limit_ssd_fallback(dir4d);
    std::filesystem::remove_all(dir4d);

    const std::string dir4h = make_temp_dir() + "-unchanged";
    test_unchanged_ssd_snapshot(dir4h);
    std::filesystem::remove_all(dir4h);

    const std::string dir4e = make_temp_dir() + "-ram-eviction";
    std::filesystem::create_directories(dir4e);
    test_ram_alloc_eviction(dir4e);
    std::filesystem::remove_all(dir4e);

    const std::string dir4f = make_temp_dir() + "-target-size-failure";
    std::filesystem::create_directories(dir4f);
    test_state_size_failure(dir4f, /* fail_draft = */ false);
    std::filesystem::remove_all(dir4f);

    const std::string dir4g = make_temp_dir() + "-draft-size-failure";
    std::filesystem::create_directories(dir4g);
    test_state_size_failure(dir4g, /* fail_draft = */ true);
    std::filesystem::remove_all(dir4g);

    const std::string dir4b = make_temp_dir() + "-fp-lora-swa";
    std::filesystem::create_directories(dir4b);
    test_fingerprint_lora_swa(dir4b);
    std::filesystem::remove_all(dir4b);

    const std::string dir5 = make_temp_dir() + "-tmp";
    std::filesystem::create_directories(dir5);
    test_tmp_cleanup(dir5);
    std::filesystem::remove_all(dir5);

    llama_backend_init();
    const std::string dir6 = make_temp_dir() + "-io-modes";
    test_io_modes(dir6);
    std::filesystem::remove_all(dir6);
    llama_backend_free();

    LOG_INF("summary: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
