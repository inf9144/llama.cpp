#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

// Borrow host pointers until synchronize(), so early source reuse fails without GPU timing assumptions.
struct deferred_backend {
    ggml_backend_ptr cpu{ggml_backend_cpu_init()};
    ggml_backend_buffer_type buft;
    ggml_backend_device device;
    ggml_backend backend = {};
    std::vector<std::function<void()>> pending;
    int synchronizations = 0;
    int writes = 0;
    int writes_pending = 0;
    int writes_max = 0;
    int event_waits = 0;
    int peer_copies = 0;
    int peer_drains = 0;

    explicit deferred_backend(bool async) {
        GGML_ASSERT(cpu);
        ggml_backend_cpu_set_n_threads(cpu.get(), 2);
        buft = *ggml_backend_get_default_buffer_type(cpu.get());
        buft.context = this;
        buft.iface.is_host = [](ggml_backend_buffer_type_t) { return false; };
        buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t type, size_t size) {
            auto * self = static_cast<deferred_backend *>(type->context);
            auto * buffer = ggml_backend_alloc_buffer(self->cpu.get(), size);
            buffer->buft = type;
            return buffer;
        };

        device = *ggml_backend_get_device(cpu.get());
        device.context = this;
        device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_GPU; };
        device.iface.get_buffer_type = [](ggml_backend_dev_t dev) {
            return &static_cast<deferred_backend *>(dev->context)->buft;
        };
        device.iface.supports_buft = [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t type) {
            return type == &static_cast<deferred_backend *>(dev->context)->buft;
        };
        device.iface.supports_op = [](ggml_backend_dev_t dev, const ggml_tensor * op) {
            return ggml_backend_supports_op(static_cast<deferred_backend *>(dev->context)->cpu.get(), op);
        };
        device.iface.event_new = [](ggml_backend_dev_t dev) {
            return new ggml_backend_event{dev, nullptr};
        };
        device.iface.event_free = [](ggml_backend_dev_t, ggml_backend_event_t event) {
            delete event;
        };
        device.iface.event_synchronize = [](ggml_backend_dev_t, ggml_backend_event_t event) {
            if (event->context) {
                ggml_backend_synchronize(static_cast<ggml_backend_t>(event->context));
            }
        };
        buft.device = &device;

        backend.device = &device;
        backend.context = this;
        backend.iface.get_name = [](ggml_backend_t) { return "deferred"; };
        backend.iface.synchronize = [](ggml_backend_t b) {
            auto * self = static_cast<deferred_backend *>(b->context);
            ++self->synchronizations;
            if (self->peer_copies > 0) {
                ++self->peer_drains;
            }
            for (auto & work : self->pending) {
                work();
            }
            self->pending.clear();
            self->writes_pending = 0;
        };
        backend.iface.graph_compute = [](ggml_backend_t b, ggml_cgraph * graph) {
            auto * self = static_cast<deferred_backend *>(b->context);
            self->pending.push_back([self, graph] {
                GGML_ASSERT(ggml_backend_graph_compute(self->cpu.get(), graph) == GGML_STATUS_SUCCESS);
            });
            return GGML_STATUS_SUCCESS;
        };
        backend.iface.get_tensor_async = [](ggml_backend_t b, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
            auto * self = static_cast<deferred_backend *>(b->context);
            self->pending.push_back([tensor, data, offset, size] {
                memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
            });
        };
        backend.iface.event_record = [](ggml_backend_t b, ggml_backend_event_t event) {
            event->context = b;
        };
        backend.iface.event_wait = [](ggml_backend_t b, ggml_backend_event_t event) {
            // This fixture uses one deferred queue; FIFO order already satisfies its own events.
            GGML_ASSERT(event->context == nullptr || event->context == b);
            ++static_cast<deferred_backend *>(b->context)->event_waits;
        };
        if (async) {
            backend.iface.set_tensor_async = [](ggml_backend_t b, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
                auto * self = static_cast<deferred_backend *>(b->context);
                ++self->writes;
                self->writes_max = std::max(self->writes_max, ++self->writes_pending);
                self->pending.push_back([tensor, data, offset, size] {
                    memcpy(static_cast<char *>(tensor->data) + offset, data, size);
                });
            };
        }
    }
};

struct copy_check {
    int calls = 0;
    float first = 0;

    static bool check(ggml_backend_t backend, const ggml_tensor *, ggml_tensor *, ggml_cgraph * graph, void * data) {
        auto & self = *static_cast<copy_check *>(data);
        auto * add = ggml_graph_node(graph, 0);
        GGML_ASSERT(add->op == GGML_OP_ADD);
        float first;
        // Order the read after queued compute, including GPU-only event waits.
        ggml_backend_tensor_get_async(backend, add->src[0], &first, 0, sizeof(first));
        ggml_backend_synchronize(backend);
        if (first != self.first) {
            fprintf(stderr, "%s scheduler callback: expected %g, got %g\n", ggml_backend_name(backend), self.first, first);
        }
        GGML_ASSERT(first == self.first);
        ++self.calls;
        return false;
    }
};

static void test_inputs(ggml_backend_t backend, bool parallel, bool computed, bool weights, int width,
        deferred_backend * deferred = nullptr, ggml_backend_t producer = nullptr) {
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    GGML_ASSERT(cpu);
    ggml_backend_cpu_set_n_threads(cpu.get(), 2);
    std::vector<ggml_backend_t> backends = {backend};
    if (producer) {
        backends.push_back(producer);
    }
    backends.push_back(cpu.get());
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends.data(), nullptr, int(backends.size()), 128, parallel, false));
    ggml_context_ptr inputs(ggml_init({8*ggml_tensor_overhead(), nullptr, true}));
    ggml_context_ptr wctx(ggml_init({ggml_tensor_overhead(), nullptr, true}));
    ggml_context_ptr ctx(ggml_init({32*ggml_tensor_overhead() + ggml_graph_overhead_custom(128, false), nullptr, true}));
    ggml_tensor * x[3];
    for (auto & tensor : x) {
        tensor = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_F32, 32, width);
        ggml_set_input(tensor);
    }
    auto * weight = ggml_new_tensor_2d(wctx.get(), GGML_TYPE_F32, 32, width);
    ggml_backend_buffer_ptr input_buffer(ggml_backend_alloc_ctx_tensors(inputs.get(), cpu.get()));
    ggml_backend_buffer_ptr weight_buffer(ggml_backend_alloc_ctx_tensors(wctx.get(), cpu.get()));
    GGML_ASSERT(input_buffer && weight_buffer);
    ggml_backend_buffer_set_usage(weight_buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> values(32*width, 1.0f);
    ggml_backend_tensor_set(weight, values.data(), 0, ggml_nbytes(weight));

    auto * first = computed ? ggml_scale(ctx.get(), x[0], 2.0f) : x[0];
    if (computed) {
        ggml_backend_sched_set_tensor_backend(sched.get(), first, producer ? producer : cpu.get());
    }
    auto * sum = ggml_add(ctx.get(), first, x[1]);
    ggml_backend_sched_set_tensor_backend(sched.get(), sum, backend);
    auto * result = ggml_add(ctx.get(), sum, x[2]);
    ggml_backend_sched_set_tensor_backend(sched.get(), result, backend);
    if (weights) {
        result = ggml_add(ctx.get(), result, weight);
        ggml_backend_sched_set_tensor_backend(sched.get(), result, backend);
    }
    ggml_set_output(result);
    auto * graph = ggml_new_graph_custom(ctx.get(), 128, false);
    ggml_build_forward_expand(graph, result);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));
    if (deferred) {
        deferred->event_waits = 0;
    }

    copy_check check;
    if (weights) {
        ggml_backend_sched_set_copy_callback(sched.get(), copy_check::check, &check);
    }
    for (int repeat = 0; repeat < 6; ++repeat) {
        for (int i = 0; i < 3; ++i) {
            std::fill(values.begin(), values.end(), float(repeat + i + 1));
            ggml_backend_tensor_set(x[i], values.data(), 0, ggml_nbytes(x[i]));
        }
        const float expected = float(repeat + 1)*(computed ? 2 : 1) + float(repeat + 2) + float(repeat + 3);
        check.first = expected;
        if (deferred) {
            deferred->synchronizations = deferred->writes = deferred->writes_max = 0;
            deferred->peer_copies = deferred->peer_drains = 0;
        }
        GGML_ASSERT(ggml_backend_sched_graph_compute_async(sched.get(), graph) == GGML_STATUS_SUCCESS);
        if (deferred) {
            const int expected = !parallel && deferred->backend.iface.set_tensor_async && width <= 512 ? (computed ? 2 : 3) : 0;
            GGML_ASSERT(deferred->writes == expected && deferred->writes_max == expected);
            if (!parallel && !weights) {
                GGML_ASSERT(deferred->synchronizations == (expected ? 2 : 1));
                if (producer) {
                    GGML_ASSERT(deferred->peer_copies == 1 && deferred->peer_drains == 0);
                }
            }
        }

        // The async scheduler must finish reading caller inputs before it returns.
        std::fill(values.begin(), values.end(), -1000.0f);
        for (auto * tensor : x) {
            ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
        }
        if (deferred && !parallel && !weights) {
            // The next call must complete this reader before reusing its input-copy storage.
            deferred->pending.push_back([result, expected] {
                const auto * data = static_cast<const float *>(result->data);
                for (int64_t i = 0; i < ggml_nelements(result); ++i) {
                    GGML_ASSERT(data[i] == expected);
                }
            });
            continue;
        }
        ggml_backend_sched_synchronize(sched.get());
        ggml_backend_tensor_get(result, values.data(), 0, ggml_nbytes(result));
        for (float value : values) {
            GGML_ASSERT(value == expected + (weights ? 1 : 0));
        }
    }
    ggml_backend_sched_synchronize(sched.get());
    const bool transfers = !ggml_backend_supports_buft(backend, ggml_backend_get_default_buffer_type(cpu.get()));
    GGML_ASSERT(check.calls == (weights && transfers ? 6 : 0));
    if (deferred && parallel && (computed || weights)) {
        GGML_ASSERT(deferred->event_waits > 0);
    }
    printf("%s scheduler inputs: width=%d parallel=%d computed=%d weights=%d async_set=%d producer=%s passed\n",
            ggml_backend_name(backend), width, parallel, computed, weights, backend->iface.set_tensor_async != nullptr,
            producer ? ggml_backend_name(producer) : "CPU");
}

int main(int argc, char ** argv) {
    if ((argc == 3 || argc == 5) && strcmp(argv[1], "--backend") == 0) {
        ggml_backend_load_all();
        ggml_backend_ptr backend(ggml_backend_init_by_name(argv[2], nullptr));
        GGML_ASSERT(backend);
        ggml_backend_ptr peer;
        if (argc == 5) {
            GGML_ASSERT(strcmp(argv[3], "--peer") == 0 && strcmp(argv[2], argv[4]) != 0);
            peer.reset(ggml_backend_init_by_name(argv[4], nullptr));
            GGML_ASSERT(peer);
        }
        for (int width : {1, 4, 1024}) {
            test_inputs(backend.get(), false, false, false, width);
            test_inputs(backend.get(), false, true, true, width);
            test_inputs(backend.get(), true, false, true, width);
            if (peer) {
                test_inputs(backend.get(), false, true, false, width, nullptr, peer.get());
                test_inputs(backend.get(), false, true, true, width, nullptr, peer.get());
            }
        }
        return 0;
    }
    GGML_ASSERT(argc == 1);
    for (bool async : {false, true}) {
        deferred_backend backend(async);
        for (bool parallel : {false, true}) {
            for (bool computed : {false, true}) {
                for (bool weights : {false, true}) {
                    test_inputs(&backend.backend, parallel, computed, weights, 4, &backend);
                }
            }
        }
        test_inputs(&backend.backend, false, false, false, 1024, &backend);
    }
    // A peer snapshot follows source compute; the destination wait must not enter host-upload completion.
    deferred_backend producer(true), consumer(true);
    consumer.backend.iface.cpy_tensor_async = [](ggml_backend_t src_backend, ggml_backend_t dst_backend,
            const ggml_tensor * src, ggml_tensor * dst) {
        if (src_backend == dst_backend || src_backend->iface.graph_compute != dst_backend->iface.graph_compute) {
            return false;
        }
        auto * source = static_cast<deferred_backend *>(src_backend->context);
        auto * target = static_cast<deferred_backend *>(dst_backend->context);
        auto snapshot = std::make_shared<std::vector<uint8_t>>(ggml_nbytes(src));
        source->pending.push_back([snapshot, src] {
            memcpy(snapshot->data(), src->data, snapshot->size());
        });
        ++target->peer_copies;
        target->pending.push_back([snapshot, src_backend, dst] {
            ggml_backend_synchronize(src_backend);
            memcpy(dst->data, snapshot->data(), snapshot->size());
        });
        return true;
    };
    test_inputs(&consumer.backend, false, true, false, 4, &consumer, &producer.backend);
}
