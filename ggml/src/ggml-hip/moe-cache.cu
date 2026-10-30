#include "../ggml-cuda/moe.cuh"
#include "../ggml-backend-impl.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace {

constexpr int max_selected = 256;

// One notification per cached layer and forward pass. The context drains it
// before another pass can reuse it. No cache weights live in this allocation.
struct alignas(64) notification {
    uint32_t published;
    int32_t count;
    int32_t miss;
    int32_t ids[max_selected];
};

struct service;
struct lookup_job {
    service * owner = nullptr;
    int32_t layer = 0;
    notification * host = nullptr;
    notification * device = nullptr;
    uint64_t * ready = nullptr;
    uint32_t serviced = 0;
#if HIP_VERSION >= 60400000
    hipStreamBatchMemOpParams wait = {};
    hipBatchMemOpNodeParams node = {};
#endif
};

struct service {
    ggml_backend_cuda_context & ctx;
    ggml_moe_resolve_t resolve;
    void * userdata;
    cudaStream_t upload = nullptr;
    cudaEvent_t ready[2] = {};
    cudaEvent_t consumed[2] = {};
    bool used[2] = {};
    bool device_lookup = false;
    std::vector<std::unique_ptr<lookup_job>> jobs;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable changed;
    std::atomic<bool> active{false};
    bool idle = true;
    bool stopping = false;

    service(ggml_backend_cuda_context & ctx, ggml_moe_resolve_t resolve, void * userdata) :
        ctx(ctx), resolve(resolve), userdata(userdata) {}

    ~service() {
        flush();
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        changed.notify_all();
        if (worker.joinable()) worker.join();
        ggml_cuda_set_device(ctx.device);
        for (auto & job : jobs) {
            if (job->ready) CUDA_CHECK(hipFree(job->ready));
            if (job->host) CUDA_CHECK(hipHostFree(job->host));
        }
        for (int i = 0; i < 2; ++i) {
            if (ready[i]) CUDA_CHECK(cudaEventDestroy(ready[i]));
            if (consumed[i]) CUDA_CHECK(cudaEventDestroy(consumed[i]));
        }
        if (upload) CUDA_CHECK(cudaStreamDestroy(upload));
    }

    void begin() {
        if (!device_lookup) return;
        {
            std::lock_guard<std::mutex> lock(mutex);
            idle = false;
            active.store(true, std::memory_order_release);
        }
        changed.notify_all();
    }

    void flush() {
        ggml_cuda_set_device(ctx.device);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        if (upload) CUDA_CHECK(cudaStreamSynchronize(upload));
        if (!worker.joinable()) return;
        active.store(false, std::memory_order_release);
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [&] { return idle; });
    }

    void run() {
        ggml_cuda_set_device(ctx.device);
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            changed.wait(lock, [&] { return stopping || active.load(std::memory_order_acquire); });
            if (stopping) return;
            lock.unlock();
            for (;;) {
                bool progress = false;
                for (auto & job : jobs) {
                    const auto version = __atomic_load_n(&job->host->published, __ATOMIC_ACQUIRE);
                    if (version == job->serviced) continue;
                    GGML_ASSERT(uint32_t(version - job->serviced) == 1);
                    const auto & request = *job->host;
                    GGML_ASSERT(request.count > 0 && request.count <= max_selected);
                    resolve(userdata, job->layer, request.ids, request.count, request.miss != 0);
                    if (request.miss) {
                        // A later request cannot arrive until this stream has
                        // uploaded the maps and released the current device wait.
                        CUDA_CHECK(hipStreamWriteValue64(upload, job->ready, 1, 0));
                    }
                    job->serviced = version;
                    progress = true;
                }
                if (!active.load(std::memory_order_acquire) && !progress) break;
                if (!progress) std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
            lock.lock();
            idle = true;
            changed.notify_all();
        }
    }

    bool init_lookup(int32_t layers) {
#if HIP_VERSION >= 60400000
        int supported = 0;
        const int physical = ggml_cuda_info().devices[ctx.device].physical_device;
        if (hipDeviceGetAttribute(&supported, hipDeviceAttributeCanUseStreamWaitValue, physical) != hipSuccess || !supported) return false;
        hipCtx_t context = nullptr;
        if (hipCtxGetCurrent(&context) != hipSuccess || !context) return false;
        for (int32_t il = 0; il < layers; ++il) {
            auto job = std::make_unique<lookup_job>();
            job->owner = this;
            job->layer = il;
            if (hipHostMalloc((void **) &job->host, sizeof(notification), hipHostMallocMapped | hipHostMallocCoherent) != hipSuccess) return false;
            memset(job->host, 0, sizeof(notification));
            jobs.push_back(std::move(job));
            auto & j = *jobs.back();
            if (hipHostGetDevicePointer((void **) &j.device, j.host, 0) != hipSuccess ||
                    hipExtMallocWithFlags((void **) &j.ready, sizeof(uint64_t), hipMallocSignalMemory) != hipSuccess) return false;
            const uint64_t one = 1;
            if (hipMemcpy(j.ready, &one, sizeof(one), hipMemcpyHostToDevice) != hipSuccess) return false;
            j.wait.operation = hipStreamMemOpWaitValue64;
            j.wait.waitValue.address = (hipDeviceptr_t) j.ready;
            j.wait.waitValue.value64 = 1;
            j.wait.waitValue.flags = hipStreamWaitValueEq;
            j.node.ctx = context;
            j.node.count = 1;
            j.node.paramArray = &j.wait;
        }
        // The probe never waits for an unsignaled value. Keep unsupported graph
        // memory operations out of dispatch, rather than discovering them live.
        hipGraph_t graph = nullptr;
        hipGraphExec_t executable = nullptr;
        hipGraphNode_t node = nullptr;
        hipError_t status = hipGraphCreate(&graph, 0);
        if (status == hipSuccess) status = hipGraphAddBatchMemOpNode(&node, graph, nullptr, 0, &jobs.front()->node);
        if (status == hipSuccess) status = hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
        if (status == hipSuccess) status = hipGraphLaunch(executable, upload);
        if (status == hipSuccess) status = hipStreamSynchronize(upload);
        if (executable) (void) hipGraphExecDestroy(executable);
        if (graph) (void) hipGraphDestroy(graph);
        if (status != hipSuccess) return false;
        device_lookup = true;
        worker = std::thread([this] { run(); });
        return true;
#else
        GGML_UNUSED(layers);
        return false;
#endif
    }
};

static __global__ void publish_lookup(const int32_t * map, const int32_t * ids, int32_t * out,
        int32_t count, int32_t experts, notification * request, uint64_t * ready) {
    const int i = threadIdx.x;
    bool miss = false;
    if (i < count) {
        const int32_t e = ids[i];
        assert(e >= 0 && e < experts);
        const int32_t slot = map[e];
        out[i] = slot;
        request->ids[i] = e;
        miss = slot < 0;
        __threadfence_system();
    }
    const int missing = __syncthreads_count(miss);
    if (i == 0) {
        request->count = count;
        request->miss = missing != 0;
        __hip_atomic_store(ready, missing ? uint64_t(0) : uint64_t(1), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        __threadfence_system();
        __hip_atomic_store(&request->published, request->published + 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

static __global__ void finish_lookup(const int32_t * map, const int32_t * ids, int32_t * out,
        int32_t count, const notification * request) {
    const int i = threadIdx.x;
    if (i < count && request->miss) out[i] = map[ids[i]];
}

} // namespace

bool ggml_hip_moe_pipeline_init(ggml_backend_t backend, int32_t layers,
        ggml_moe_resolve_t resolve, void * userdata, ggml_moe_pipeline * out) {
    auto & ctx = *(ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(ctx.device);
    const auto & info = ggml_cuda_info().devices[ctx.device];
    if (layers <= 0 || !resolve || !GGML_CUDA_CC_IS_RDNA3_0(info.cc) || info.warp_size != 32) return false;
    auto engine = std::make_unique<service>(ctx, resolve, userdata);
    CUDA_CHECK(cudaStreamCreateWithFlags(&engine->upload, cudaStreamNonBlocking));
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&engine->ready[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&engine->consumed[i], cudaEventDisableTiming));
    }
    if (!engine->init_lookup(layers)) {
        (void) hipGetLastError();
        GGML_LOG_WARN("%s: %s device routing unavailable; retain host routing, enable transfer pipeline\n", __func__, ctx.name.c_str());
    }
    *out = {
        engine.get(), engine->device_lookup,
        [](void * p) {
            auto & e = *(service *) p;
            e.flush();
#ifdef USE_CUDA_GRAPH
            e.ctx.cuda_graphs.clear();
#endif
            delete &e;
        },
        [](void * p) { ((service *) p)->begin(); },
        [](void * p) { ((service *) p)->flush(); },
        [](void * p, ggml_tensor * dst, const void * src, size_t offset, size_t bytes) {
            auto & e = *(service *) p;
            ggml_cuda_set_device(e.ctx.device);
            ggml_cuda_set_tensor_async_stream(e.ctx, dst, src, offset, bytes, e.upload);
        },
        [](void * p, int32_t lane) {
            auto & e = *(service *) p;
            GGML_ASSERT(lane >= 0 && lane < 2);
            if (e.used[lane]) CUDA_CHECK(cudaStreamWaitEvent(e.upload, e.consumed[lane], 0));
        },
        [](void * p, int32_t lane, ggml_moe_packet * packet) {
            auto & e = *(service *) p;
            CUDA_CHECK(cudaEventRecord(e.ready[lane], e.upload));
            packet->ready = (void *) e.ready[lane];
            packet->consumed = (void *) e.consumed[lane];
            e.used[lane] = true;
        },
        [](void * p, int32_t il) -> void * {
            auto & e = *(service *) p;
            GGML_ASSERT(e.device_lookup && il >= 0 && size_t(il) < e.jobs.size());
            return e.jobs[il].get();
        },
    };
    engine.release();
    return true;
}

void ggml_hip_moe_cache_lookup(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto & job = *(lookup_job *) ggml_moe_get_lookup_job(dst);
    GGML_ASSERT(job.owner->device_lookup && &job.owner->ctx == &ctx);
    const auto * map = (const int32_t *) dst->src[0]->data;
    const auto * ids = (const int32_t *) dst->src[1]->data;
    auto * slots = (int32_t *) dst->data;
    const int count = ggml_nelements(dst);
    GGML_ASSERT(count > 0 && count <= max_selected);
    const auto stream = ctx.stream();
    publish_lookup<<<1, max_selected, 0, stream>>>(map, ids, slots, count, dst->src[0]->ne[1], job.device, job.ready);
#if HIP_VERSION >= 60400000
    hipStreamCaptureStatus status;
    hipGraph_t graph;
    const hipGraphNode_t * dependencies;
    size_t n_dependencies;
    CUDA_CHECK(hipStreamGetCaptureInfo_v2(stream, &status, nullptr, &graph, &dependencies, &n_dependencies));
    if (status == hipStreamCaptureStatusActive) {
        hipGraphNode_t wait;
        CUDA_CHECK(hipGraphAddBatchMemOpNode(&wait, graph, dependencies, n_dependencies, &job.node));
        CUDA_CHECK(hipStreamUpdateCaptureDependencies(stream, &wait, 1, hipStreamSetCaptureDependencies));
    } else
#endif
    {
        CUDA_CHECK(hipStreamWaitValue64(stream, job.ready, 1, hipStreamWaitValueEq, UINT64_MAX));
    }
    finish_lookup<<<1, max_selected, 0, stream>>>(map, ids, slots, count, job.device);
    CUDA_CHECK(cudaGetLastError());
}
