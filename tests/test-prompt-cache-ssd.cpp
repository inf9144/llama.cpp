#include "server-prompt-cache-ssd.h"

#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
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

// test the file format round-trip
static void test_roundtrip(const std::string & dir) {
    const auto fp = make_fp();
    server_prompt_cache_ssd cache(dir, 0);
    CHECK(cache.init(fp), "init");

    const std::vector<llama_token> toks = {1, 2, 3, 4, 5, 6, 7, 8};
    auto state = make_state(toks, 100, 50);

    CHECK(cache.save(state), "save");
    CHECK(cache.n_entries() == 1, "n_entries == 1");
    CHECK(cache.size() > 0, "size > 0");

    // verify the file exists
    const auto files = std::filesystem::directory_iterator(dir);
    int n_files = 0;
    for (const auto & p : files) {
        if (p.is_regular_file() && p.path().filename().string().find(".lsc") != std::string::npos) {
            n_files++;
        }
    }
    CHECK(n_files == 1, "one .lsc file");

    // find the entry and verify the tokens
    int lcp = -1;
    const auto * e = cache.find_best(server_tokens(llama_tokens(toks), false), lcp);
    CHECK(e != nullptr, "find_best finds the entry");
    CHECK(lcp == 8, "lcp == 8");

    // verify the state can be read back
    server_prompt prompt;
    server_prompt_data data;
    const std::string path = dir + "/" + e->key + ".lsc";
    // use a fresh cache to read the state (simulates a restart)
    server_prompt_cache_ssd cache2(dir, 0);
    CHECK(cache2.init(fp), "init2");
    CHECK(cache2.n_entries() == 1, "n_entries2 == 1");
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
    for (const auto & p : std::filesystem::directory_iterator(dir)) {
        if (p.is_regular_file() && p.path().filename().string().find(".lsc") != std::string::npos) {
            n_files++;
        }
    }
    CHECK(n_files == 1, "file kept on disk");

    // switching back to the original model should see the entry again
    server_prompt_cache_ssd cache3(dir, 0);
    CHECK(cache3.init(fp), "init3");
    CHECK(cache3.n_entries() == 1, "entry visible again");
}

// test corrupt file handling
static void test_corrupt(const std::string & dir) {
    const auto fp = make_fp();
    server_prompt_cache_ssd cache(dir, 0);
    CHECK(cache.init(fp), "init");

    // create a corrupt file (bad magic)
    const std::string corrupt_path = dir + "/deadbeefdeadbeefdeadbeefdeadbeef.lsc";
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
    for (const auto & p : std::filesystem::directory_iterator(dir)) {
        if (p.is_regular_file() && p.path().filename().string().find(".lsc") != std::string::npos) {
            valid_path = p.path().string();
            break;
        }
    }
    CHECK(!valid_path.empty(), "found .lsc file");

    // truncate the file
    const auto file_size = std::filesystem::file_size(valid_path);
    {
        std::ofstream f(valid_path, std::ios::binary | std::ios::trunc);
        std::ifstream in(valid_path, std::ios::binary);
        std::vector<char> data(file_size);
        in.read(data.data(), data.size());
        f.write(data.data(), data.size() / 2);  // write only half
    }

    // re-init, the truncated file should be removed
    server_prompt_cache_ssd cache3(dir, 0);
    CHECK(cache3.init(fp), "init3");
    CHECK(!std::filesystem::exists(valid_path), "truncated file removed");
}

// test temp file cleanup
static void test_tmp_cleanup(const std::string & dir) {
    const auto fp = make_fp();

    // create a temp file (simulates a crashed write)
    const std::string tmp_path = dir + "/deadbeefdeadbeefdeadbeefdeadbeef.lsc.tmp";
    {
        std::ofstream f(tmp_path, std::ios::binary);
        f.write("partial", 7);
    }

    server_prompt_cache_ssd cache(dir, 0);
    CHECK(cache.init(fp), "init");
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

    const std::string dir3 = make_temp_dir() + "-fp";
    std::filesystem::create_directories(dir3);
    test_fingerprint_mismatch(dir3);
    std::filesystem::remove_all(dir3);

    const std::string dir4 = make_temp_dir() + "-corrupt";
    std::filesystem::create_directories(dir4);
    test_corrupt(dir4);
    std::filesystem::remove_all(dir4);

    const std::string dir5 = make_temp_dir() + "-tmp";
    std::filesystem::create_directories(dir5);
    test_tmp_cleanup(dir5);
    std::filesystem::remove_all(dir5);

    LOG_INF("summary: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
