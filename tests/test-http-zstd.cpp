#include "../vendor/cpp-httplib/httplib.h"

#include <zstd.h>

#include <cstdio>
#include <string>
#include <thread>

#ifndef CPPHTTPLIB_ZSTD_SUPPORT
#error "test-http-zstd requires CPPHTTPLIB_ZSTD_SUPPORT"
#endif

static int fail(const char * message) {
    std::fprintf(stderr, "test-http-zstd: %s\n", message);
    return 1;
}

int main() {
    const std::string payload =
        R"({\"model\":\"qwen3.8-27b\",\"input\":[{\"role\":\"user\",\"content\":\"zstd request body\"}],\"stream\":true})";

    httplib::Server server;
    bool body_matches = false;
    server.Post("/v1/responses", [&](const httplib::Request & req, httplib::Response & res) {
        body_matches = req.body == payload;
        res.status = body_matches ? 200 : 400;
        res.set_content(body_matches ? "ok" : "body mismatch", "text/plain");
    });

    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        return fail("failed to bind loopback server");
    }

    std::thread worker([&]() {
        server.listen_after_bind();
    });

    std::string compressed(ZSTD_compressBound(payload.size()), '\0');
    const size_t compressed_size = ZSTD_compress(
        compressed.data(), compressed.size(), payload.data(), payload.size(), 1);
    if (ZSTD_isError(compressed_size)) {
        server.stop();
        worker.join();
        return fail(ZSTD_getErrorName(compressed_size));
    }
    compressed.resize(compressed_size);

    httplib::Client client("127.0.0.1", port);
    const httplib::Headers headers = {{"Content-Encoding", "zstd"}};
    const auto result = client.Post("/v1/responses", headers, compressed, "application/json");

    server.stop();
    worker.join();

    if (!result) {
        return fail("compressed POST failed");
    }
    if (result->status != 200) {
        return fail("handler did not receive the decompressed request body");
    }
    if (!body_matches) {
        return fail("request body mismatch after zstd decompression");
    }

    std::puts("test-http-zstd: passed");
    return 0;
}
