#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

static void check(ggml_tensor * tensor, const std::vector<float> & expected) {
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(tensor, actual.data(), 0, ggml_nbytes(tensor));
    GGML_ASSERT(memcmp(actual.data(), expected.data(), ggml_nbytes(tensor)) == 0);
}

static ggml_context * make_context() {
    return ggml_init({ 16 * ggml_tensor_overhead(), nullptr, true });
}

static void test_transfers(ggml_backend_t backend, bool benchmark) {
    constexpr size_t bytes = 4 * 1024 * 1024;
    constexpr size_t count = bytes / sizeof(float);
    constexpr size_t view_offset = 256;
    ggml_context * ctx = make_context();
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, count);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, count);
    ggml_tensor * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, count);
    ggml_tensor * view = ggml_view_1d(ctx, a, count - view_offset / sizeof(float), view_offset);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_buffer_type_t host_type = ggml_backend_dev_host_buffer_type(dev);
    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    GGML_ASSERT(cpu);
    if (!host_type) {
        host_type = ggml_backend_get_default_buffer_type(cpu);
    }
    ggml_context * host_ctx = make_context();
    ggml_tensor * source = ggml_new_tensor_1d(host_ctx, GGML_TYPE_F32, count);
    ggml_backend_buffer_t host_buffer = ggml_backend_alloc_ctx_tensors_from_buft(host_ctx, host_type);
    GGML_ASSERT(host_buffer);

    std::vector<float> input(count), zeros(count, 0.0f), expected(count);
    for (size_t i = 0; i < count; ++i) {
        input[i] = float((i * 17 + 13) % 4096);
    }
    ggml_backend_tensor_set(source, input.data(), 0, bytes);
    fprintf(stderr, "transfer source: %s\n", ggml_backend_buffer_name(host_buffer));

    // Read with a GPU copy so a coherent CPU read alone cannot hide missing host-write visibility.
    for (size_t size : { size_t(256), size_t(128 * 1024), size_t(128 * 1024 + 4), size_t(2 * 1024 * 1024) }) {
        for (size_t offset : { size_t(0), size_t(68) }) {
            for (const void * data : { static_cast<const void *>(input.data()), static_cast<const void *>(source->data) }) {
                expected = zeros;
                ggml_backend_tensor_set(a, zeros.data(), 0, bytes);
                ggml_backend_synchronize(backend);
                ggml_backend_tensor_set_async(backend, view, data, offset, size);
                memcpy(reinterpret_cast<char *>(expected.data()) + view_offset + offset, data, size);
                ggml_backend_tensor_copy_async(backend, backend, a, b);
                ggml_backend_synchronize(backend);
                check(b, expected);
            }
        }
    }

    for (size_t width : { size_t(1024), size_t(256 * 1024 + 12) }) {
        for (size_t source_gap : { size_t(0), size_t(64) }) {
            for (size_t dest_gap : { size_t(0), size_t(128) }) {
                for (const void * data : { static_cast<const void *>(input.data()), static_cast<const void *>(source->data) }) {
                    constexpr size_t rows = 3;
                    constexpr size_t offset = 68;
                    expected = zeros;
                    ggml_backend_tensor_set(a, zeros.data(), 0, bytes);
                    ggml_backend_synchronize(backend);
                    ggml_backend_tensor_set_2d_async(backend, view, data, offset, width, rows, width + dest_gap, width + source_gap);
                    for (size_t row = 0; row < rows; ++row) {
                        memcpy(reinterpret_cast<char *>(expected.data()) + view_offset + offset + row * (width + dest_gap),
                               static_cast<const char *>(data) + row * (width + source_gap), width);
                    }
                    ggml_backend_tensor_copy_async(backend, backend, a, b);
                    ggml_backend_synchronize(backend);
                    check(b, expected);
                }
            }
        }
    }

    // Exercise the pinned CPU-source cpy callback above the former 128 KiB limit.
    ggml_backend_tensor_copy_async(cpu, backend, source, a);
    ggml_backend_tensor_copy_async(backend, backend, a, b);
    ggml_backend_synchronize(backend);
    check(b, input);
    expected = input;

    ggml_tensor * source_offset_view = ggml_view_1d(host_ctx, source, count / 2, 64);
    ggml_tensor * dest_offset_view = ggml_view_1d(ctx, a, count / 2, 128);
    ggml_backend_view_init(source_offset_view);
    ggml_backend_view_init(dest_offset_view);
    ggml_backend_tensor_copy_async(cpu, backend, source_offset_view, dest_offset_view);
    ggml_backend_synchronize(backend);
    memcpy(reinterpret_cast<char *>(expected.data()) + 128, static_cast<const char *>(source->data) + 64, bytes / 2);
    check(a, expected);

    ggml_backend_tensor_set(a, zeros.data(), 0, bytes);
    ggml_backend_tensor_copy_async(backend, backend, a, b);
    ggml_backend_tensor_copy_async(cpu, backend, source, a);
    ggml_backend_synchronize(backend);
    check(b, zeros);
    check(a, input);
    expected = input;

    // Reuse four cache slots. Each old GPU read must finish before the slot is overwritten.
    for (size_t round = 0; round < 24; ++round) {
        const size_t slot_size = 512 * 1024;
        const size_t offset = (round % 4) * slot_size;
        const size_t source_offset = (round % 7) * 64;
        ggml_backend_tensor_copy_async(backend, backend, a, b);
        ggml_backend_tensor_set_async(backend, a, static_cast<const char *>(source->data) + source_offset, offset, slot_size);
        ggml_backend_synchronize(backend);
        check(b, expected);
        memcpy(reinterpret_cast<char *>(expected.data()) + offset, static_cast<const char *>(source->data) + source_offset, slot_size);
        check(a, expected);
    }

    // A different idle backend must also see a live context on this device as busy.
    ggml_backend_t other = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(other);
    ggml_backend_tensor_copy_async(backend, backend, a, b);
    ggml_backend_tensor_set_async(other, c, source->data, 0, bytes);
    ggml_backend_synchronize(other);
    ggml_backend_synchronize(backend);
    check(b, expected);
    check(c, input);

    ggml_backend_event_t event = ggml_backend_event_new(dev);
    if (event) {
        ggml_backend_tensor_copy_async(backend, backend, a, b);
        ggml_backend_event_record(event, backend);
        ggml_backend_event_wait(other, event);
        ggml_backend_tensor_set_async(other, a, source->data, 0, bytes);
        ggml_backend_synchronize(other);
        ggml_backend_synchronize(backend);
        check(b, expected);
        check(a, input);
        ggml_backend_event_free(event);
    }

    // Concurrent contexts use disjoint tensors and keep the pinned source alive through each sync.
    const auto worker = [&](ggml_backend_t owner, ggml_tensor * dst) {
        for (int i = 0; i < 16; ++i) {
            ggml_backend_tensor_set_async(owner, dst, source->data, 0, bytes);
            ggml_backend_synchronize(owner);
            check(dst, input);
        }
    };
    std::thread first(worker, backend, a);
    std::thread second(worker, other, c);
    first.join();
    second.join();

    if (benchmark) {
        fprintf(stdout, "path,bytes,burst,copies,wall_us,MiB_per_second\n");
        for (size_t size : { size_t(4096), size_t(256 * 1024), size_t(1024 * 1024), bytes }) {
            ggml_tensor * source_view = ggml_view_1d(host_ctx, source, size / sizeof(float), 0);
            ggml_tensor * dest_view = ggml_view_1d(ctx, a, size / sizeof(float), 0);
            ggml_backend_view_init(source_view);
            ggml_backend_view_init(dest_view);
            for (int path = 0; path < 3; ++path) {
                for (int burst : { 1, 8 }) {
                    constexpr int repeats = 32;
                    const int64_t start = ggml_time_us();
                    for (int repeat = 0; repeat < repeats; ++repeat) {
                        for (int copy = 0; copy < burst; ++copy) {
                            if (path == 2) {
                                ggml_backend_tensor_copy_async(cpu, backend, source_view, dest_view);
                            } else {
                                ggml_backend_tensor_set_async(backend, a, path == 0 ? input.data() : source->data, 0, size);
                            }
                        }
                        ggml_backend_synchronize(backend);
                    }
                    const int64_t elapsed = ggml_time_us() - start;
                    fprintf(stdout, "%s,%zu,%d,%d,%lld,%.2f\n", path == 0 ? "set-pageable" : path == 1 ? "set-pinned" : "cpy-pinned",
                            size, burst, repeats * burst, static_cast<long long>(elapsed),
                            double(size) * repeats * burst * 1e6 / (1024.0 * 1024.0 * double(elapsed)));
                }
            }
        }
    }

    ggml_backend_free(other);
    ggml_backend_free(cpu);
    ggml_backend_buffer_free(host_buffer);
    ggml_free(host_ctx);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    const char * name = argc > 1 ? argv[1] : "Vulkan0";
    ggml_backend_t backend = ggml_backend_init_by_name(name, nullptr);
    if (!backend) {
        fprintf(stderr, "%s unavailable; skipping transfer tests\n", name);
        return 77;
    }
    test_transfers(backend, argc > 2 && strcmp(argv[2], "--bench") == 0);
    ggml_backend_free(backend);
    fprintf(stderr, "%s: asynchronous transfer tests passed\n", name);
    return 0;
}
