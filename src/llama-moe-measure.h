#pragma once

#ifdef LLAMA_MOE_MEASURE
#include "ggml-backend.h"
#include "ggml.h"
#include <atomic>
#include <cstdio>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace llama_moe_measure {
inline std::atomic<uint64_t> sequence{0};
inline thread_local uint64_t graph = 0;
inline thread_local uintptr_t context = 0;
inline thread_local std::string pending;
inline void flush() {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    fwrite(pending.data(), 1, pending.size(), stderr);
    pending.clear();
}
inline thread_local int tokens = 0, graph_type = 0;
inline std::string quote(const char * value) {
    std::string result = "\"";
    for (const unsigned char * p = (const unsigned char *) value; *p; ++p) {
        if (*p == '"' || *p == '\\') { result += '\\'; result += char(*p); }
        else if (*p < 32) { char tmp[7]; snprintf(tmp, sizeof(tmp), "\\u%04x", *p); result += tmp; }
        else { result += char(*p); }
    }
    return result + '"';
}
struct record {
    std::ostringstream out;
    record(const char * kind, ggml_backend_t backend = nullptr, const ggml_tensor * tensor = nullptr) {
        out << "MOE_MEASURE\t{\"schema\":1,\"kind\":" << quote(kind)
            << ",\"us\":" << ggml_time_us() << ",\"graph\":" << graph << ",\"context\":" << context
            << ",\"tokens\":" << tokens << ",\"graph_type\":" << graph_type;
        if (backend) { out << ",\"backend\":" << quote(ggml_backend_name(backend)); }
        if (tensor) { out << ",\"tensor\":" << quote(tensor->name) << ",\"type\":" << quote(ggml_type_name(tensor->type)); }
    }
    ~record() {
        out << "}\n";
        pending += out.str();
    }
};
inline void routing(ggml_backend_t backend, const ggml_tensor * weight, const int32_t * ids, size_t count,
                    int experts, int64_t wait_us, const int32_t * slots = nullptr) {
    std::vector<size_t> histogram(experts, 0);
    for (size_t i = 0; i < count; ++i) { GGML_ASSERT(ids[i] >= 0 && ids[i] < experts); ++histogram[ids[i]]; }
    record r("routing", backend, weight);
    r.out << ",\"wait_us\":" << wait_us << ",\"counts\":[";
    for (int i = 0; i < experts; ++i) { if (i) { r.out << ','; } r.out << histogram[i]; }
    r.out << ']';
    if (slots) {
        r.out << ",\"resident\":[";
        for (int i = 0; i < experts; ++i) { if (i) { r.out << ','; } r.out << (slots[i] >= 0 ? 1 : 0); }
        r.out << ']';
    }
}
inline void transfer(const char * kind, ggml_backend_t backend, const ggml_tensor * weight,
                     int64_t first, int64_t count, size_t bytes, int64_t enqueue_us) {
    record r(kind, backend, weight);
    r.out << ",\"first\":" << first << ",\"count\":" << count << ",\"bytes\":" << bytes
          << ",\"expert_bytes\":" << weight->nb[2] << ",\"enqueue_us\":" << enqueue_us;
}
}
#define MOE_MEASURE(...) do { __VA_ARGS__; } while (false)
#else
#define MOE_MEASURE(...) do {} while (false)
#endif
