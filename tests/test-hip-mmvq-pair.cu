#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-cuda/mmvq-pair.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

template <int ncols>
static __global__ void reference(const void * weights, const block_q8_1 * input, float * dst, int k, int rows, int stride) {
#if defined(RDNA3_0)
    float sum[ncols] = {};
    const int row = blockIdx.x;
    for (int b = threadIdx.x/(QI6_K/2); b < k/QK_K; b += 2*WARP_SIZE/QI6_K) {
        const int q = 2*(threadIdx.x%(QI6_K/2));
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            sum[j] += vec_dot_q6_K_q8_1_vdr2(weights, input + j*stride + b*(QK_K/QK8_1), row*(k/QK_K) + b, q);
        }
    }
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        sum[j] = warp_reduce_sum<WARP_SIZE>(sum[j]);
        if (threadIdx.x == 0) dst[j*rows + row] = sum[j];
    }
#else
    GGML_UNUSED_VARS(weights, input, dst, k, rows, stride);
    NO_DEVICE_CODE;
#endif
}

template <typename F>
static hipGraphExec_t capture(F launch, hipStream_t stream) {
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    CUDA_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    for (int i = 0; i < 32; ++i) launch();
    CUDA_CHECK(hipStreamEndCapture(stream, &graph));
    CUDA_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    CUDA_CHECK(hipGraphDestroy(graph));
    return exec;
}

static double time_graph(hipGraphExec_t graph, hipStream_t stream) {
    hipEvent_t a, b;
    CUDA_CHECK(hipEventCreate(&a));
    CUDA_CHECK(hipEventCreate(&b));
    CUDA_CHECK(hipEventRecord(a, stream));
    CUDA_CHECK(hipGraphLaunch(graph, stream));
    CUDA_CHECK(hipEventRecord(b, stream));
    CUDA_CHECK(hipEventSynchronize(b));
    float ms = 0;
    CUDA_CHECK(hipEventElapsedTime(&ms, a, b));
    CUDA_CHECK(hipEventDestroy(a));
    CUDA_CHECK(hipEventDestroy(b));
    return 1000.0*ms/32;
}

template <int ncols>
static void run(ggml_backend_t backend, int k, int rows_a, int rows_b, bool view, bool bench) {
    ggml_context_ptr weights(ggml_init({4*ggml_tensor_overhead(), nullptr, true}));
    auto make_weight = [&](int rows) {
        auto * parent = ggml_new_tensor_2d(weights.get(), GGML_TYPE_Q6_K, k, rows + 2);
        return ggml_view_2d(weights.get(), parent, k, rows, parent->nb[1], parent->nb[1]);
    };
    auto * a = make_weight(rows_a);
    auto * b = make_weight(rows_b);
    ggml_backend_buffer_ptr wb(ggml_backend_alloc_ctx_tensors(weights.get(), backend));
    GGML_ASSERT(wb);
    ggml_backend_buffer_set_usage(wb.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_clear(wb.get(), 0xff);
    ggml_context_ptr ctx(ggml_init({8*ggml_tensor_overhead() + 3*ggml_graph_overhead(), nullptr, true}));
    auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, ncols);
    auto * y_a = ggml_mul_mat(ctx.get(), a, x);
    auto * y_b = ggml_mul_mat(ctx.get(), b, x);
    auto * first = ggml_new_graph(ctx.get());
    auto * second = ggml_new_graph(ctx.get());
    auto * pair = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(first, y_a);
    ggml_build_forward_expand(second, y_b);
    ggml_build_forward_expand(pair, view ? ggml_reshape_2d(ctx.get(), y_a, rows_a, ncols) : y_a);
    ggml_build_forward_expand(pair, y_b);
    ggml_backend_buffer_ptr cb(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(cb);
    uint32_t seed = 42;
    auto random = [&]() { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed; };
    for (auto * weight : {a, b}) {
        std::vector<block_q6_K> host(ggml_nbytes(weight)/sizeof(block_q6_K));
        for (auto & block : host) {
            for (auto & q : block.ql) q = random();
            for (auto & q : block.qh) q = random();
            for (auto & s : block.scales) s = int(random()%255) - 127;
            block.d = ggml_fp32_to_fp16((int(random()%33) - 16)/1024.0f);
        }
        ggml_backend_tensor_set(weight, host.data(), 0, ggml_nbytes(weight));
    }
    const int stride = GGML_PAD(k, MATRIX_ROW_PADDING)/QK8_1;
    const size_t na = size_t(ncols)*rows_a, nb = size_t(ncols)*rows_b, guard = 64;
    ggml_backend_buffer_ptr qb(ggml_backend_alloc_buffer(backend, ncols*stride*sizeof(block_q8_1)));
    ggml_backend_buffer_ptr rb(ggml_backend_alloc_buffer(backend, (na + nb + 2*guard)*sizeof(float)));
    ggml_backend_buffer_ptr ab(ggml_backend_alloc_buffer(backend, (na + nb + 2*guard)*sizeof(float)));
    GGML_ASSERT(qb && rb && ab);
    auto * quantized = (block_q8_1 *) ggml_backend_buffer_get_base(qb.get());
    auto * old = (float *) ggml_backend_buffer_get_base(rb.get());
    auto * result = (float *) ggml_backend_buffer_get_base(ab.get());
    hipStream_t stream;
    CUDA_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    auto baseline = [&]() {
        reference<ncols><<<rows_a, WARP_SIZE, 0, stream>>>(a->data, quantized, old, k, rows_a, stride);
        reference<ncols><<<rows_b, WARP_SIZE, 0, stream>>>(b->data, quantized, old + na + guard, k, rows_b, stride);
    };
    auto candidate = [&]() {
        mul_mat_vec_q6_k_pair<ncols><<<rows_a + rows_b, WARP_SIZE, 0, stream>>>(a->data, b->data, quantized,
                result, result + na + guard, k, rows_a, rows_b, stride);
    };
    std::vector<float> xf(size_t(k)*ncols), expected(na + nb + 2*guard), actual(expected.size()), automatic(na + nb), unfused(na + nb);
    std::vector<block_q8_1> q(ncols*stride);
    for (int pass = 0; pass < 4; ++pass) {
        for (int t = 0; t < ncols; ++t) for (int i = 0; i < stride; ++i) {
            auto & block = q[t*stride + i];
            const bool live = i < k/QK8_1 && pass != 1;
            const float scale = live ? std::ldexp(1.0f, pass == 2 ? -12 : pass == 3 ? 3 : -5) : 0.0f;
            int sum = 0;
            for (int j = 0; j < QK8_1; ++j) {
                block.qs[j] = live ? (j == 0 ? 127 : int(random()%255) - 127) : 0;
                sum += block.qs[j];
                if (i < k/QK8_1) xf[(t*k + i*QK8_1) + j] = block.qs[j]*scale;
            }
            const float ds[] = {scale, sum*scale};
            ggml_fp32_to_fp16_row(ds, (ggml_fp16_t *) &block.ds, 2);
        }
        ggml_backend_tensor_set(x, xf.data(), 0, ggml_nbytes(x));
        GGML_ASSERT(ggml_backend_graph_compute(backend, first) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(ggml_backend_graph_compute(backend, second) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(y_a, unfused.data(), 0, na*sizeof(float));
        ggml_backend_tensor_get(y_b, unfused.data() + na, 0, nb*sizeof(float));
        GGML_ASSERT(ggml_backend_graph_compute(backend, pair) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(y_a, automatic.data(), 0, na*sizeof(float));
        ggml_backend_tensor_get(y_b, automatic.data() + na, 0, nb*sizeof(float));
        CUDA_CHECK(hipMemcpyAsync(quantized, q.data(), q.size()*sizeof(block_q8_1), hipMemcpyHostToDevice, stream));
        CUDA_CHECK(hipMemsetAsync(old, 0xff, expected.size()*sizeof(float), stream));
        CUDA_CHECK(hipMemsetAsync(result, 0xff, actual.size()*sizeof(float), stream));
        baseline(); candidate();
        CUDA_CHECK(hipGetLastError());
        CUDA_CHECK(hipStreamSynchronize(stream));
        CUDA_CHECK(hipMemcpy(expected.data(), old, expected.size()*sizeof(float), hipMemcpyDeviceToHost));
        CUDA_CHECK(hipMemcpy(actual.data(), result, actual.size()*sizeof(float), hipMemcpyDeviceToHost));
        for (size_t i = 0; i < expected.size(); ++i) {
            const bool padding = (i >= na && i < na + guard) || i >= na + guard + nb;
            if (padding) {
                GGML_ASSERT(std::isnan(expected[i]) && std::isnan(actual[i]));
                continue;
            }
            const size_t j = i < na ? i : i - guard;
            GGML_ASSERT(std::isfinite(expected[i]) && std::isfinite(actual[i]) && std::isfinite(automatic[j]));
            GGML_ASSERT(std::memcmp(&expected[i], &actual[i], sizeof(float)) == 0);
            GGML_ASSERT(std::memcmp(&expected[i], &automatic[j], sizeof(float)) == 0);
            GGML_ASSERT(std::memcmp(&unfused[j], &automatic[j], sizeof(float)) == 0);
        }
    }
    std::printf("%s Q6_K pair K=%d M=%d+%d tokens=%d view=%d: four bitwise passes, automatic dispatch and guards passed\n",
            ggml_backend_name(backend), k, rows_a, rows_b, ncols, view);
    if (bench) {
        const auto old_graph = capture(baseline, stream), new_graph = capture(candidate, stream);
        std::vector<double> times[2];
        for (int r = 0; r < 8; ++r) for (int i = 0; i < 2; ++i) {
            const int j = r%2 ? 1 - i : i;
            times[j].push_back(time_graph(j ? new_graph : old_graph, stream));
        }
        for (auto & values : times) std::sort(values.begin(), values.end());
        const double before = (times[0][3] + times[0][4])/2, after = (times[1][3] + times[1][4])/2;
        std::printf("%s BENCH Q6_K pair K=%d M=%d+%d tokens=%d matmul_only_original_us=%.3f matmul_only_pair_us=%.3f speedup=%.3f\n",
                ggml_backend_name(backend), k, rows_a, rows_b, ncols, before, after, before/after);
        CUDA_CHECK(hipGraphExecDestroy(old_graph));
        CUDA_CHECK(hipGraphExecDestroy(new_graph));
    }
    std::fflush(stdout);
    CUDA_CHECK(hipStreamDestroy(stream));
}

int main(int argc, char ** argv) {
    const char * name = "ROCm0";
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) name = argv[++i];
        else if (std::strcmp(argv[i], "--bench") == 0) bench = true;
        else { std::fprintf(stderr, "Usage: %s [--backend ROCm0] [--bench]\n", argv[0]); return 1; }
    }
    ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_name(name);
    if (!dev) { std::fprintf(stderr, "Missing backend %s\n", name); return 1; }
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    GGML_ASSERT(backend && std::strncmp(ggml_backend_name(backend.get()), "ROCm", 4) == 0);
    ggml_backend_buffer_ptr probe(ggml_backend_alloc_buffer(backend.get(), 1));
    GGML_ASSERT(probe);
    int device;
    hipDeviceProp_t props;
    CUDA_CHECK(hipGetDevice(&device));
    CUDA_CHECK(hipGetDeviceProperties(&props, device));
    if (std::strncmp(props.gcnArchName, "gfx110", 6) != 0 || props.warpSize != WARP_SIZE) {
        std::fprintf(stderr, "This test requires RDNA3 wave32\n"); return 1;
    }
    run<2>(backend.get(), 768, 129, 65, true, false);
    run<3>(backend.get(), 256, 17, 33, false, false);
    run<4>(backend.get(), 2560, 129, 65, true, false);
    run<2>(backend.get(), 2560, 10240, 6144, false, bench);
    run<3>(backend.get(), 2560, 10240, 6144, true, bench);
    run<4>(backend.get(), 2560, 10240, 6144, true, bench);
    return 0;
}
