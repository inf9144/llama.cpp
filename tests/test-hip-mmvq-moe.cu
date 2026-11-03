#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-cuda/mmvq-moe.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

template <ggml_type type>
static __device__ __forceinline__ float reference_dot(const void * w, const block_q8_1 * y, int b, int q) {
    if constexpr (type == GGML_TYPE_IQ2_S)   return vec_dot_iq2_s_q8_1(w, y, b, q);
    if constexpr (type == GGML_TYPE_IQ3_XXS) return vec_dot_iq3_xxs_q8_1(w, y, b, q);
    if constexpr (type == GGML_TYPE_IQ3_S)   return vec_dot_iq3_s_q8_1(w, y, b, q);
    if constexpr (type == GGML_TYPE_IQ4_NL)  return vec_dot_iq4_nl_q8_1(w, y, b, q);
    if constexpr (type == GGML_TYPE_IQ4_XS)  return vec_dot_iq4_xs_q8_1(w, y, b, q);
    if constexpr (type == GGML_TYPE_Q2_0)    return vec_dot_q2_0_q8_1(w, y, b, q);
}

// Keep the original two-row, one-wave-per-token MoE accumulation as the reference.
template <ggml_type type, bool fused>
__launch_bounds__((type == GGML_TYPE_Q2_0 ? 8 : type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_IQ4_XS ? 6 : 4)*ggml_cuda_get_physical_warp_size(), 1)
static __global__ void reference_moe(const void * weights, const block_q8_1 * input, const int32_t * ids,
        ggml_cuda_mm_fusion_args_device fusion, float * dst, int k, int m, int used, int channels, int id_stride,
        int weight_row, int weight_expert, int input_token, int input_channel) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = type == GGML_TYPE_Q2_0 ? VDR_Q2_0_Q8_1_MMVQ : type == GGML_TYPE_IQ4_XS ? VDR_IQ4_XS_Q8_1_MMVQ : 2;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const bool shared = fused && fusion.shared_up && blockIdx.y == uint32_t(used);
    const int t = threadIdx.y, rank = shared ? 0 : blockIdx.y, row = 2*blockIdx.x;
    const int expert = shared ? 0 : ids[t*id_stride + rank];
    const void * gate = shared ? fusion.shared_gate : fusion.gate;
    if (shared) {
        weights = fusion.shared_up;
        dst = fusion.shared_dst;
    }
    const auto * y = input + t*input_token + (rank%channels)*input_channel;
    float sum[2] = {}, g[2] = {};
    for (int b = threadIdx.x/(qi/vdr); b < k/qk; b += vdr*warp_size/qi) {
        const int q = vdr*(threadIdx.x%(qi/vdr));
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (row + r < m) {
                const int index = expert*weight_expert + (row + r)*weight_row + b;
                sum[r] += reference_dot<type>(weights, y + b*(qk/QK8_1), index, q);
                if constexpr (fused) g[r] += reference_dot<type>(gate, y + b*(qk/QK8_1), index, q);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        sum[r] = warp_reduce_sum<warp_size>(sum[r]);
        if constexpr (fused) g[r] = warp_reduce_sum<warp_size>(g[r]);
    }
    if (threadIdx.x < 2 && row + int(threadIdx.x) < m) {
        float v = sum[threadIdx.x];
        if constexpr (fused) {
            const float value = g[threadIdx.x];
            switch (fusion.glu_op) {
                case GGML_GLU_OP_SWIGLU:       v *= ggml_cuda_op_silu_single(value); break;
                case GGML_GLU_OP_GEGLU:        v *= ggml_cuda_op_gelu_single(value); break;
                case GGML_GLU_OP_SWIGLU_OAI:   v = ggml_cuda_op_swiglu_oai_single(value, v); break;
                case GGML_GLU_OP_SWIGLU_CLAMP: v = ggml_cuda_op_swiglu_clamp_single(value, v, fusion.glu_limit); break;
                default:                      v *= value; break;
            }
        }
        dst[(shared ? t : t*used + rank)*m + row + threadIdx.x] = v;
    }
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

template <ggml_type type>
static void run(ggml_backend_t backend, int k, int m, int used, bool broadcast, bool fused, bool shared, bool bench) {
    constexpr int slots = 65, guard = 64;
    const int channels = broadcast ? 1 : used;
    const int id_stride = used + 3;
    const int weight_row = k/ggml_blck_size(type) + 2;
    const int weight_expert = (m + 2)*weight_row;
    const size_t block_size = ggml_type_size(type), prefix = 7*block_size;
    std::vector<uint8_t> w(prefix + slots*weight_expert*block_size), gate(w.size());
    uint32_t seed = 42;
    auto random = [&]() { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed; };
    for (auto * bytes : {&w, &gate}) {
        for (auto & v : *bytes) v = random();
        for (size_t i = 0; i < bytes->size(); i += block_size) {
            const float scale = (int(random()%17) - 8)/512.0f;
            if constexpr (type == GGML_TYPE_Q2_0) {
                std::memcpy(bytes->data() + i, &scale, sizeof(scale));
            } else {
                const auto h = ggml_fp32_to_fp16(scale);
                std::memcpy(bytes->data() + i, &h, sizeof(h));
            }
        }
    }
    const int input_channel = GGML_PAD(k, 512)/QK8_1, input_token = channels*input_channel;
    std::vector<block_q8_1> x(4*input_token);
    for (auto & b : x) {
        const float ds[] = {1.0f/32, 0.0f};
        ggml_fp32_to_fp16_row(ds, (ggml_fp16_t *) &b.ds, 2);
        for (auto & q : b.qs) q = int(random()%255) - 127;
    }
    auto allocate = [&](size_t n) {
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_buffer(backend, n));
        GGML_ASSERT(buffer);
        return buffer;
    };
    auto wb = allocate(w.size()), gb = allocate(gate.size()), xb = allocate(x.size()*sizeof(block_q8_1));
    auto ib = allocate(4*id_stride*sizeof(int32_t)), desc = allocate((4*used + 1)*sizeof(mmvq_moe_group));
    const size_t n = 4*used*m, ns = shared ? 4*m : 0;
    auto old_buffer = allocate((n + ns + guard)*sizeof(float)), new_buffer = allocate((n + ns + guard)*sizeof(float));
    const auto * weights = (const char *) ggml_backend_buffer_get_base(wb.get()) + prefix;
    const auto * gates = (const char *) ggml_backend_buffer_get_base(gb.get()) + prefix;
    auto * input = (block_q8_1 *) ggml_backend_buffer_get_base(xb.get());
    auto * ids = (int32_t *) ggml_backend_buffer_get_base(ib.get());
    auto * groups = (mmvq_moe_group *) ggml_backend_buffer_get_base(desc.get());
    auto * before = (float *) ggml_backend_buffer_get_base(old_buffer.get());
    auto * after = (float *) ggml_backend_buffer_get_base(new_buffer.get());
    hipStream_t stream;
    CUDA_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    CUDA_CHECK(hipMemcpyAsync(ggml_backend_buffer_get_base(wb.get()), w.data(), w.size(), hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemcpyAsync(ggml_backend_buffer_get_base(gb.get()), gate.data(), gate.size(), hipMemcpyHostToDevice, stream));
    CUDA_CHECK(hipMemcpyAsync(input, x.data(), x.size()*sizeof(block_q8_1), hipMemcpyHostToDevice, stream));
    std::vector<int32_t> routing(4*id_stride, -1);
    std::vector<float> expected(n + ns + guard), actual(expected.size());
    std::vector<mmvq_moe_group> plan(4*used + 1);
    const ggml_glu_op glu[] = {GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU, GGML_GLU_OP_SWIGLU_OAI, GGML_GLU_OP_SWIGLU_CLAMP};
    for (int pattern = 0; pattern < 5; ++pattern) {
        for (int t = 0; t < 4; ++t) for (int r = 0; r < used; ++r) {
            int e = pattern == 0 ? t*used + r : pattern == 1 ? (r + 3*t)%used : pattern == 2 ? (r + 6*t)%28 : pattern == 3 ? r%3 : 0;
            routing[t*id_stride + r] = (e*37 + 11)%slots;
        }
        CUDA_CHECK(hipMemcpyAsync(ids, routing.data(), routing.size()*sizeof(int32_t), hipMemcpyHostToDevice, stream));
        CUDA_CHECK(hipMemsetAsync(before, 0xff, expected.size()*sizeof(float), stream));
        CUDA_CHECK(hipMemsetAsync(after, 0xff, actual.size()*sizeof(float), stream));
        CUDA_CHECK(hipMemsetAsync(groups, 0xff, plan.size()*sizeof(mmvq_moe_group), stream));
        ggml_cuda_mm_fusion_args_device fusion = {};
        fusion.gate = fused ? gates : nullptr;
        fusion.glu_op = glu[pattern%4];
        fusion.glu_limit = 7.0f;
        if (shared) {
            fusion.shared_up = weights;
            fusion.shared_gate = gates;
            fusion.shared_stride_col_dst = m;
        }
        auto baseline = [&]() {
            fusion.shared_dst = before + n;
            if (fused) {
                reference_moe<type, true><<<dim3((m + 1)/2, used + shared), dim3(WARP_SIZE, 4), 0, stream>>>(weights, input, ids, fusion, before,
                        k, m, used, channels, id_stride, weight_row, weight_expert, input_token, input_channel);
            } else {
                reference_moe<type, false><<<dim3((m + 1)/2, used + shared), dim3(WARP_SIZE, 4), 0, stream>>>(weights, input, ids, fusion, before,
                        k, m, used, channels, id_stride, weight_row, weight_expert, input_token, input_channel);
            }
        };
        auto select = [&]() { mmvq_moe_group_ids<<<1, 64, 0, stream>>>(ids, used, id_stride, groups); };
        auto matmul = [&]() {
            fusion.shared_dst = after + n;
            mul_mat_vec_q_moe_grouped_launch<type>(weights, input, groups, fusion, after, k, m, used, channels,
                    weight_row, weight_expert, input_token, input_channel, used*m, m, WARP_SIZE, stream);
        };
        auto candidate = [&]() { select(); matmul(); };
        baseline(); candidate();
        CUDA_CHECK(hipGetLastError());
        CUDA_CHECK(hipStreamSynchronize(stream));
        CUDA_CHECK(hipMemcpy(expected.data(), before, expected.size()*sizeof(float), hipMemcpyDeviceToHost));
        CUDA_CHECK(hipMemcpy(actual.data(), after, actual.size()*sizeof(float), hipMemcpyDeviceToHost));
        CUDA_CHECK(hipMemcpy(plan.data(), groups, plan.size()*sizeof(mmvq_moe_group), hipMemcpyDeviceToHost));
        std::vector<int> visits(4*used, 0);
        const int count = plan[4*used].count;
        GGML_ASSERT(count > 0 && count <= 4*used);
        for (int g = 0; g < count; ++g) {
            GGML_ASSERT(plan[g].count > 0 && plan[g].count <= 4);
            int tokens = 0;
            for (int j = 0; j < plan[g].count; ++j) {
                const int row = plan[g].rows[j];
                GGML_ASSERT(row >= 0 && row < 4*used);
                GGML_ASSERT(!(tokens & (1 << (row/used))));
                tokens |= 1 << (row/used);
                GGML_ASSERT(plan[g].expert == routing[(row/used)*id_stride + row%used]);
                ++visits[row];
            }
        }
        GGML_ASSERT(std::all_of(visits.begin(), visits.end(), [](int n) { return n == 1; }));
        size_t changed = 0;
        double max_abs = 0;
        for (size_t i = 0; i < n + ns; ++i) {
            GGML_ASSERT(std::isfinite(expected[i]) && std::isfinite(actual[i]));
            changed += std::memcmp(&expected[i], &actual[i], sizeof(float)) != 0;
            max_abs = std::max(max_abs, std::abs(double(expected[i]) - actual[i]));
        }
        std::printf("%s %s K=%d M=%d used=%d broadcast=%d fused=%d shared=%d pattern=%d groups=%d changed=%zu max_abs=%.9g\n",
                ggml_backend_name(backend), ggml_type_name(type), k, m, used, broadcast, fused, shared, pattern, count, changed, max_abs);
        std::fflush(stdout);
        GGML_ASSERT(changed == 0);
        for (size_t i = n + ns; i < actual.size(); ++i) GGML_ASSERT(std::isnan(expected[i]) && std::isnan(actual[i]));
        if (bench && pattern <= 2) {
            const auto a = capture(baseline, stream), b = capture(candidate, stream);
            const auto c = capture(select, stream), d = capture(matmul, stream);
            std::vector<double> times[4];
            for (int repeat = 0; repeat < 8; ++repeat) {
                for (int j = 0; j < 4; ++j) {
                    const int index = repeat%2 ? 3 - j : j;
                    const hipGraphExec_t graphs[] = {a, b, c, d};
                    times[index].push_back(time_graph(graphs[index], stream));
                }
            }
            double median[4];
            for (int i = 0; i < 4; ++i) {
                std::sort(times[i].begin(), times[i].end());
                median[i] = (times[i][3] + times[i][4])/2;
            }
            std::printf("%s BENCH %s K=%d M=%d pattern=%d original_us=%.3f grouped_total_us=%.3f select_us=%.3f matmul_us=%.3f speedup=%.3f\n",
                    ggml_backend_name(backend), ggml_type_name(type), k, m, pattern, median[0], median[1], median[2], median[3], median[0]/median[1]);
            for (auto graph : {a, b, c, d}) CUDA_CHECK(hipGraphExecDestroy(graph));
        }
    }
    CUDA_CHECK(hipStreamDestroy(stream));
}

template <ggml_type type>
static void cases(ggml_backend_t backend, bool bench) {
    run<type>(backend, 256, 129, 16, false, true, false, false);
    run<type>(backend, 768, 129, 10, true, false, false, false);
    run<type>(backend, 2560, 512, 10, true, true, true, bench);
    run<type>(backend, 512, 2560, 10, false, false, false, bench);
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
    cases<GGML_TYPE_IQ2_S>(backend.get(), bench);
    cases<GGML_TYPE_IQ3_XXS>(backend.get(), bench);
    cases<GGML_TYPE_IQ3_S>(backend.get(), bench);
    cases<GGML_TYPE_IQ4_NL>(backend.get(), bench);
    cases<GGML_TYPE_IQ4_XS>(backend.get(), bench);
    cases<GGML_TYPE_Q2_0>(backend.get(), bench);
    return 0;
}
