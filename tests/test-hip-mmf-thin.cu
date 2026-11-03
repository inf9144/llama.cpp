#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-cuda/mmf.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

static __global__ void reference_scale(float * dst, int n, float bias) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = 0.25f*dst[i] + bias;
    }
}

static __global__ void reference_silu(float * dst, int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = ggml_cuda_op_silu_single(dst[i]);
    }
}

template <typename F>
static hipGraphExec_t capture(F launch, hipStream_t stream) {
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    CUDA_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    for (int i = 0; i < 64; ++i) {
        launch();
    }
    CUDA_CHECK(hipStreamEndCapture(stream, &graph));
    CUDA_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(hipGraphDestroy(graph));
    return exec;
}

static double time_graph(hipGraphExec_t graph, hipStream_t stream) {
    hipEvent_t begin, end;
    CUDA_CHECK(hipEventCreate(&begin));
    CUDA_CHECK(hipEventCreate(&end));
    CUDA_CHECK(hipEventRecord(begin, stream));
    CUDA_CHECK(hipGraphLaunch(graph, stream));
    CUDA_CHECK(hipEventRecord(end, stream));
    CUDA_CHECK(hipEventSynchronize(end));
    float ms = 0;
    CUDA_CHECK(hipEventElapsedTime(&ms, begin, end));
    CUDA_CHECK(hipEventDestroy(begin));
    CUDA_CHECK(hipEventDestroy(end));
    return 1000.0*ms/64;
}

template <int tokens>
static void run(ggml_backend_t backend, int k, int m, bool bench, bool silu, float bias = 0.0f, bool retain_mat = false) {
    ggml_context_ptr ctx_weights(ggml_init({ggml_tensor_overhead(), nullptr, true}));
    ggml_context_ptr ctx(ggml_init({8*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true}));
    auto * weights = ggml_new_tensor_2d(ctx_weights.get(), GGML_TYPE_BF16, k, m);
    auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, tokens);
    ggml_set_input(input);
    auto * result = ggml_mul_mat(ctx.get(), weights, input);
    if (retain_mat) {
        ggml_set_output(result);
    }
    if (silu) {
        result = ggml_silu(ctx.get(), ggml_scale_bias(ctx.get(), result, 0.25f, bias));
    }
    auto * graph = ggml_new_graph(ctx.get());
    ggml_set_output(result);
    ggml_build_forward_expand(graph, result);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx_weights.get(), backend));
    GGML_ASSERT(buffer);
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(&backend, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, true));
    GGML_ASSERT(sched && ggml_backend_sched_alloc_graph(sched.get(), graph));

    std::vector<float> wf(size_t(k)*m), xf(size_t(k)*tokens);
    std::vector<ggml_bf16_t> w(wf.size());
    uint32_t seed = 42;
    auto random = [&]() {
        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
        return std::ldexp((int(seed & 65535) - 32768)/32768.0f, int((seed >> 16) % 11) - 5);
    };
    for (auto & v : wf) {
        v = random()/64.0f;
    }
    ggml_fp32_to_bf16_row(wf.data(), w.data(), wf.size());
    ggml_backend_tensor_set(weights, w.data(), 0, ggml_nbytes(weights));

    const size_t n = size_t(m)*tokens;
    constexpr int guard = 64;
    float * reference = nullptr;
    float * actual = nullptr;
    float * scratch = nullptr;
    CUDA_CHECK(hipMalloc(&reference, (n + guard)*sizeof(float)));
    CUDA_CHECK(hipMalloc(&actual, (n + guard)*sizeof(float)));
    CUDA_CHECK(hipMalloc(&scratch, (MMF_THIN_NWARPS*n + guard)*sizeof(float)));
    hipStream_t stream;
    CUDA_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    auto baseline = [&]() {
        const int shared_bytes = MMF_THIN_NWARPS*16*(WARP_SIZE + 4)*sizeof(nv_bfloat162);
        mul_mat_f<nv_bfloat162, MMF_ROWS_PER_BLOCK, tokens, MMF_THIN_NWARPS, false>
            <<<dim3(m/MMF_ROWS_PER_BLOCK), dim3(WARP_SIZE, MMF_THIN_NWARPS), shared_bytes, stream>>>(
                (const nv_bfloat162 *) weights->data, (const float *) input->data, nullptr, reference,
                k/2, tokens, 1, k/2, k/2, m, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0);
        if (silu) {
            reference_scale<<<(n + 255)/256, 256, 0, stream>>>(reference, n, bias);
            reference_silu<<<(n + 255)/256, 256, 0, stream>>>(reference, n);
        }
    };
    auto candidate = [&]() {
        mul_mat_f_thin_cuda<tokens>((const nv_bfloat162 *) weights->data, (const float *) input->data,
                scratch, actual, k/2, m, stream, silu, 0.25f, bias);
    };
    std::vector<float> expected(n + guard), output(n + guard), automatic(n), partial(MMF_THIN_NWARPS*n + guard);
    std::printf("%s BF16 HC down K=%d M=%d tokens=%d silu=%d bias=%g retain_mat=%d: ",
            ggml_backend_name(backend), k, m, tokens, silu, bias, retain_mat);
    std::fflush(stdout);
    for (int pass = 0; pass < 4; ++pass) {
        for (auto & v : xf) {
            v = pass == 1 ? 0.0f : random();
            if (pass == 3) {
                v = std::ldexp(v, -120);
            }
        }
        ggml_backend_tensor_set(input, xf.data(), 0, ggml_nbytes(input));
        CUDA_CHECK(hipMemsetAsync(reference, 0xff, (n + guard)*sizeof(float), stream));
        CUDA_CHECK(hipMemsetAsync(actual, 0xff, (n + guard)*sizeof(float), stream));
        CUDA_CHECK(hipMemsetAsync(scratch, 0xff, partial.size()*sizeof(float), stream));
        baseline();
        candidate();
        CUDA_CHECK(hipGetLastError());
        CUDA_CHECK(hipStreamSynchronize(stream));
        CUDA_CHECK(hipMemcpy(expected.data(), reference, expected.size()*sizeof(float), hipMemcpyDeviceToHost));
        CUDA_CHECK(hipMemcpy(output.data(), actual, output.size()*sizeof(float), hipMemcpyDeviceToHost));
        CUDA_CHECK(hipMemcpy(partial.data(), scratch, partial.size()*sizeof(float), hipMemcpyDeviceToHost));
        GGML_ASSERT(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(result, automatic.data(), 0, ggml_nbytes(result));
        double max_abs = 0;
        size_t changed = 0, automatic_changed = 0;
        for (size_t i = 0; i < n; ++i) {
            GGML_ASSERT(std::isfinite(expected[i]) && std::isfinite(output[i]) && std::isfinite(automatic[i]));
            max_abs = std::max(max_abs, double(std::abs(expected[i] - output[i])));
            changed += std::memcmp(&expected[i], &output[i], sizeof(float)) != 0;
            automatic_changed += std::memcmp(&expected[i], &automatic[i], sizeof(float)) != 0;
        }
        if (changed || automatic_changed) {
            std::printf("pass=%d changed=%zu automatic_changed=%zu max_abs=%.9g\n", pass, changed, automatic_changed, max_abs);
            std::fflush(stdout);
        }
        GGML_ASSERT(changed == 0);
        GGML_ASSERT(automatic_changed == 0);
        for (size_t i = n; i < expected.size(); ++i) {
            GGML_ASSERT(std::isnan(expected[i]) && std::isnan(output[i]));
        }
        for (size_t i = MMF_THIN_NWARPS*n; i < partial.size(); ++i) {
            GGML_ASSERT(std::isnan(partial[i]));
        }
    }
    std::puts("four bitwise passes and guards passed");

    if (bench) {
        for (auto & v : xf) {
            v = std::clamp(random(), -1.0f, 1.0f);
        }
        ggml_backend_tensor_set(input, xf.data(), 0, ggml_nbytes(input));
        const auto old_graph = capture(baseline, stream);
        const auto new_graph = capture(candidate, stream);
        std::vector<double> before, after;
        for (int i = 0; i < 10; ++i) {
            if (i % 2 == 0) {
                before.push_back(time_graph(old_graph, stream));
                after.push_back(time_graph(new_graph, stream));
            } else {
                after.push_back(time_graph(new_graph, stream));
                before.push_back(time_graph(old_graph, stream));
            }
        }
        std::sort(before.begin(), before.end());
        std::sort(after.begin(), after.end());
        const double old_us = (before[4] + before[5])/2;
        const double new_us = (after[4] + after[5])/2;
        std::printf("%s BENCH K=%d M=%d tokens=%d silu=%d original_us=%.3f split_us=%.3f speedup=%.3f scratch_KiB=%.1f\n",
                ggml_backend_name(backend), k, m, tokens, silu, old_us, new_us, old_us/new_us, MMF_THIN_NWARPS*n*sizeof(float)/1024.0);
        CUDA_CHECK(hipGraphExecDestroy(old_graph));
        CUDA_CHECK(hipGraphExecDestroy(new_graph));
    }
    CUDA_CHECK(hipStreamDestroy(stream));
    CUDA_CHECK(hipFree(scratch));
    CUDA_CHECK(hipFree(actual));
    CUDA_CHECK(hipFree(reference));
}

int main(int argc, char ** argv) {
    const char * name = "ROCm0";
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            name = argv[++i];
        } else if (std::strcmp(argv[i], "--bench") == 0) {
            bench = true;
        } else {
            std::fprintf(stderr, "Usage: %s [--backend ROCm0] [--bench]\n", argv[0]);
            return 1;
        }
    }
    ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_name(name);
    if (!dev) {
        std::fprintf(stderr, "Missing backend %s\n", name);
        return 1;
    }
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    GGML_ASSERT(backend && std::strncmp(ggml_backend_name(backend.get()), "ROCm", 4) == 0);
    // Select the backend's physical device before using native HIP allocations and streams.
    ggml_backend_buffer_ptr probe(ggml_backend_alloc_buffer(backend.get(), 1));
    GGML_ASSERT(probe);
    int device;
    hipDeviceProp_t props;
    CUDA_CHECK(hipGetDevice(&device));
    CUDA_CHECK(hipGetDeviceProperties(&props, device));
    if (std::strncmp(props.gcnArchName, "gfx110", 6) != 0 || props.warpSize != WARP_SIZE) {
        std::fprintf(stderr, "This test requires RDNA3 wave32\n");
        return 1;
    }
    for (const auto shape : {std::pair<int, int>{8192, 32}, {10240, 320}, {16384, 512}, {65536, 32}}) {
        for (bool silu : {false, true}) {
            run<4>(backend.get(), shape.first, shape.second, bench, silu);
            run<5>(backend.get(), shape.first, shape.second, bench, silu);
            run<6>(backend.get(), shape.first, shape.second, bench, silu);
            run<7>(backend.get(), shape.first, shape.second, bench, silu);
            run<8>(backend.get(), shape.first, shape.second, bench, silu);
        }
    }
    run<4>(backend.get(), 8192, 32, false, true, 0.125f);
    run<4>(backend.get(), 8192, 32, false, true, -0.0f, true);
    return 0;
}
