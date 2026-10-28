#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static ggml_context_ptr context() {
    return ggml_context_ptr(ggml_init({2*1024*1024, nullptr, true}));
}

template <typename T>
static void set_rows(ggml_tensor * t, const std::vector<T> & values) {
    std::vector<uint8_t> bytes(ggml_nbytes(t), 0);
    for (int64_t row = 0; row < t->ne[1]; ++row) {
        for (int64_t col = 0; col < t->ne[0]; ++col) {
            memcpy(bytes.data() + row*t->nb[1] + col*t->nb[0], &values[row*t->ne[0] + col], sizeof(T));
        }
    }
    ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
}

static void random_tensor(ggml_tensor * t, std::mt19937 & gen) {
    std::vector<float> data(ggml_nelements(t));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto & v : data) { v = dist(gen); }
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
    } else {
        std::vector<uint8_t> quant(ggml_nbytes(t));
        ggml_quantize_chunk(t->type, data.data(), quant.data(), 0, ggml_nrows(t), t->ne[0], nullptr);
        ggml_backend_tensor_set(t, quant.data(), 0, quant.size());
    }
}

struct qsa_graph {
    ggml_context_ptr ctx = context();
    ggml_gallocr_ptr alloc;
    ggml_cgraph * graph;
    ggml_tensor * ranked;
    ggml_tensor * selection;
    ggml_tensor * out;

    qsa_graph(ggml_backend_t backend, bool sparse, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
            ggml_tensor * top, ggml_tensor * score, ggml_tensor * pools, ggml_tensor * tail, bool rank = false) {
        const int64_t kv = k->ne[1];
        if (rank) {
            // Give each alternative its own first node, hence its own HIP graph-cache key.
            GGML_ASSERT(q->view_src);
            q = ggml_view_4d(ctx.get(), q->view_src, q->ne[0], q->ne[1], q->ne[2], q->ne[3], q->nb[1], q->nb[2], q->nb[3], q->view_offs);
        }
        ranked = rank ? ggml_top_k(ctx.get(), score, top->ne[0]) : top;
        if (sparse) {
            selection = ggml_qsa_select(ctx.get(), ranked, score, pools, tail, kv);
            auto * idx = ggml_reshape_4d(ctx.get(), selection, selection->ne[0], q->ne[1], 1, q->ne[3]);
            out = ggml_flash_attn_sparse(ctx.get(), q, k, v, idx, 1.0f/16.0f);
        } else {
            auto * zeros = ggml_fill(ctx.get(), ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, kv), 0.0f);
            selection = ggml_top_k_mask(ctx.get(), zeros, ranked, score, pools, tail);
            auto * mask = ggml_reshape_4d(ctx.get(), selection, kv, q->ne[1], 1, q->ne[3]);
            out = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, 1.0f/16.0f, 0.0f, 0.0f);
            ggml_prec_set_acc(out, GGML_PREC_F32);
        }
        graph = ggml_new_graph_custom(ctx.get(), 64, false);
        ggml_build_forward_expand(graph, out);
        alloc.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
    }

    void allocate() {
        GGML_ASSERT(ggml_gallocr_alloc_graph(alloc.get(), graph));
    }
};

static double timed(ggml_backend_t backend, ggml_cgraph * graph) {
    for (int i = 0; i < 2; ++i) { GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS); }
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; ++i) { GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS); }
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count()/10;
}

static double quantile(std::vector<double> samples, double q) {
    std::sort(samples.begin(), samples.end());
    return samples[std::min(samples.size() - 1, size_t(std::ceil(q*samples.size())) - 1)];
}

static void measure_crossover(ggml_backend_t backend, qsa_graph & dense, qsa_graph & sparse, int kv, int width, bool permuted) {
    std::array<qsa_graph *, 2> branches = {&dense, &sparse};
    std::array<std::array<ggml_cgraph *, 4>, 2> graphs;
    std::array<std::vector<uint8_t>, 2> reference_selection, reference_output;
    for (int b = 0; b < 2; ++b) {
        auto * branch = branches[b];
        reference_selection[b].resize(ggml_nbytes(branch->selection));
        reference_output[b].resize(ggml_nbytes(branch->out));
        ggml_backend_tensor_get(branch->selection, reference_selection[b].data(), 0, reference_selection[b].size());
        ggml_backend_tensor_get(branch->out, reference_output[b].data(), 0, reference_output[b].size());
        for (int stage = 0; stage < 3; ++stage) {
            graphs[b][stage] = ggml_new_graph_custom(branch->ctx.get(), 64, false);
        }
        ggml_build_forward_expand(graphs[b][0], branch->ranked);
        for (int i = 0; i < ggml_graph_n_nodes(branch->graph); ++i) {
            auto * node = ggml_graph_node(branch->graph, i);
            if (node != branch->ranked && node != branch->out) {
                ggml_graph_add_node(graphs[b][1], node);
            }
        }
        ggml_graph_add_node(graphs[b][2], branch->out);
        graphs[b][3] = branch->graph;
    }
    std::array<std::array<std::vector<double>, 4>, 2> samples;
    // Keep stages separate: partial and complete graphs can share the HIP graph-cache key.
    // Rank before selection, then attention: the allocator can reuse Top-K storage for attention output.
    for (int stage = 0; stage < 4; ++stage) {
        for (int b = 0; b < 2; ++b) {
            for (int i = 0; i < 5; ++i) {
                GGML_ASSERT(ggml_backend_graph_compute(backend, graphs[b][stage]) == GGML_STATUS_SUCCESS);
            }
        }
        for (int round = 0; round < 21; ++round) {
            for (int j = 0; j < 2; ++j) {
                const int b = (round + j)%2;
                const auto start = std::chrono::steady_clock::now();
                for (int i = 0; i < 5; ++i) {
                    GGML_ASSERT(ggml_backend_graph_compute(backend, graphs[b][stage]) == GGML_STATUS_SUCCESS);
                }
                samples[b][stage].push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count()/5);
            }
        }
        for (int b = 0; b < 2; ++b) {
            auto check = [&](ggml_tensor * tensor, const std::vector<uint8_t> & expected) {
                std::vector<uint8_t> actual(expected.size());
                ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
                if (actual != expected) {
                    GGML_ABORT("component graph mismatch: kv=%d width=%d path=%s stage=%d", kv, width, b ? "sparse" : "dense", stage);
                }
            };
            if (stage == 1 || stage == 3) { check(branches[b]->selection, reference_selection[b]); }
            if (stage == 2 || stage == 3) { check(branches[b]->out, reference_output[b]); }
        }
    }
    for (int b = 0; b < 2; ++b) {
        printf("CROSS backend=%s kv=%d width=%d layout=%s path=%s topk_us=%.3f select_us=%.3f attention_us=%.3f qsa_graph_us=%.3f qsa_p95_us=%.3f compute_buffer=%zu selection_bytes=%zu exact=1\n",
            ggml_backend_name(backend), kv, width, permuted ? "permuted" : "linear", b ? "sparse" : "dense",
            quantile(samples[b][0], 0.5), quantile(samples[b][1], 0.5), quantile(samples[b][2], 0.5),
            quantile(samples[b][3], 0.5), quantile(samples[b][3], 0.95), ggml_gallocr_get_buffer_size(branches[b]->alloc.get(), 0),
            ggml_nbytes(branches[b]->selection));
    }
    printf("CROSS_RESULT backend=%s kv=%d width=%d layout=%s sparse_speedup=%.4f\n", ggml_backend_name(backend), kv, width,
        permuted ? "permuted" : "linear", quantile(samples[0][3], 0.5)/quantile(samples[1][3], 0.5));
    fflush(stdout);
}

static void run_case(ggml_backend_t backend, int kv, int width, int top_count, int streams, bool strided, bool bench,
        bool crossover = false, bool permuted = false) {
    auto inputs = context();
    const int rows = width*streams;
    const int n_pool = kv/4;
    auto input = [&](ggml_type type, int cols, int rows) {
        ggml_tensor * t = ggml_new_tensor_2d(inputs.get(), type, cols + (strided ? 3 : 0), rows);
        return strided ? ggml_view_2d(inputs.get(), t, cols, rows, t->nb[1], 0) : t;
    };
    ggml_tensor * top = input(GGML_TYPE_I32, top_count, rows);
    ggml_tensor * score = input(GGML_TYPE_F32, n_pool, rows);
    ggml_tensor * pools = input(GGML_TYPE_I32, 4, n_pool);
    ggml_tensor * tail = input(GGML_TYPE_I32, 3, rows);
    ggml_tensor * qr = ggml_new_tensor_4d(inputs.get(), GGML_TYPE_F32, 256, 24, width, streams);
    ggml_tensor * kr = ggml_new_tensor_4d(inputs.get(), GGML_TYPE_Q8_0, 256, 2, kv, streams);
    ggml_tensor * vr = ggml_dup_tensor(inputs.get(), kr);
    ggml_tensor * q = ggml_permute(inputs.get(), qr, 0, 2, 1, 3);
    ggml_tensor * k = ggml_permute(inputs.get(), kr, 0, 2, 1, 3);
    ggml_tensor * v = ggml_permute(inputs.get(), vr, 0, 2, 1, 3);
    ggml_backend_buffer_ptr input_buffer(ggml_backend_alloc_ctx_tensors(inputs.get(), backend));
    GGML_ASSERT(input_buffer);
    std::mt19937 gen(0x515341);
    for (auto * t : {qr, kr, vr}) { random_tensor(t, gen); }
    std::vector<int32_t> top_data(top_count*rows);
    std::vector<float> score_data(n_pool*rows);
    std::vector<int32_t> pool_data(4*n_pool);
    std::vector<int32_t> tail_data(3*rows);
    std::vector<int32_t> cells(kv);
    for (int i = 0; i < kv; ++i) { cells[i] = i; }
    if (permuted) { std::shuffle(cells.begin(), cells.end(), gen); }
    for (int p = 0; p < n_pool; ++p) {
        for (int i = 0; i < 4; ++i) {
            // Padded pools, sentinel/negative members, duplicate members and overlapping pools.
            pool_data[4*p + i] = bench ? cells[4*p + i] : p%13 == 0 || (p%19 == 0 && i == 3) ? kv :
                p%17 == 0 ? -1 : (4*p + (i == 3 && p%7 == 0 ? 0 : i))%kv;
        }
    }
    set_rows(pools, pool_data);
    qsa_graph dense(backend, false, q, k, v, top, score, pools, tail, crossover);
    qsa_graph sparse(backend, true, q, k, v, top, score, pools, tail, crossover);
    const bool supported = ggml_backend_supports_op(backend, sparse.out) && ggml_backend_supports_op(backend, sparse.selection);
    if (!supported) {
        GGML_ASSERT(!crossover);
        printf("kv=%d width=%d: dense fallback (sparse dispatch not eligible)\n", kv, width);
        return;
    }
    dense.allocate();
    sparse.allocate();
    double max_abs = 0.0;
    double max_nmse = 0.0;
    for (int state = 0; state < (bench ? 1 : 4); ++state) {
        for (int row = 0; row < rows; ++row) {
            const int visible = kv - width + row + 1;
            for (int p = 0; p < n_pool; ++p) {
                score_data[row*n_pool + p] = crossover ? (p < visible/4 ? float(gen()%0x1000000) : -INFINITY) :
                    !bench && (p + row + state)%5 == 0 ? -INFINITY : float(p%19);
            }
            for (int i = 0; i < top_count; ++i) {
                // Unordered, repeated picks; some are invalid or invisible for this query.
                top_data[row*top_count + i] = bench ? (i*37 + row*3)%n_pool :
                    i%29 == 0 ? n_pool : i%31 == 0 ? -1 : ((i%127)*37 + row*3 + state)%n_pool;
            }
            tail_data[3*row + 0] = (row*3 + state)%kv;
            tail_data[3*row + 1] = (row*3 + state)%kv;
            tail_data[3*row + 2] = kv;
            if (crossover) {
                for (int j = 0; j < 3; ++j) {
                    const int logical = visible/4*4 + j;
                    tail_data[3*row + j] = logical < visible ? cells[logical] : kv;
                }
            }
        }
        set_rows(top, top_data); set_rows(score, score_data); set_rows(tail, tail_data);
        GGML_ASSERT(ggml_backend_graph_compute(backend, dense.graph) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(ggml_backend_graph_compute(backend, sparse.graph) == GGML_STATUS_SUCCESS);
        std::vector<ggml_fp16_t> mask(kv*rows);
        std::vector<int32_t> selected(sparse.selection->ne[0]*rows);
        ggml_backend_tensor_get(dense.selection, mask.data(), 0, ggml_nbytes(dense.selection));
        ggml_backend_tensor_get(sparse.selection, selected.data(), 0, ggml_nbytes(sparse.selection));
        const int n_sel = sparse.selection->ne[0];
        for (int row = 0; row < rows; ++row) {
            int next = 0;
            for (int cell = 0; cell < kv; ++cell) {
                if (ggml_fp16_to_fp32(mask[row*kv + cell]) == 0.0f) {
                    GGML_ASSERT(next < n_sel && selected[row*n_sel + next++] == cell);
                }
            }
            while (next < n_sel) { GGML_ASSERT(selected[row*n_sel + next++] == -1); }
        }
        std::vector<float> a(ggml_nelements(dense.out)), b(a.size());
        ggml_backend_tensor_get(dense.out, a.data(), 0, ggml_nbytes(dense.out));
        ggml_backend_tensor_get(sparse.out, b.data(), 0, ggml_nbytes(sparse.out));
        double error = 0.0, energy = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            GGML_ASSERT(std::isfinite(a[i]) && std::isfinite(b[i]));
            max_abs = std::max(max_abs, double(fabsf(a[i] - b[i])));
            error += double(a[i] - b[i])*(a[i] - b[i]);
            energy += double(a[i])*a[i];
        }
        max_nmse = std::max(max_nmse, error/std::max(energy, 1e-30));
        GGML_ASSERT(max_nmse <= 5e-4);

        // Re-evaluate each verify row as decode. Both paths must use the same KV reduction.
        if (!bench && width <= 8) {
            for (int row = 0; row < width; ++row) {
                auto single = context();
                auto * dec = ggml_new_graph_custom(single.get(), 64, false);
                auto * q1 = ggml_view_4d(single.get(), q, 256, 1, 24, streams, q->nb[1], q->nb[2], q->nb[3], row*q->nb[1]);
                // Copy one row per stream into a contiguous index tensor.
                auto * idx = ggml_new_tensor_4d(single.get(), GGML_TYPE_I32, n_sel, 1, 1, streams);
                ggml_backend_buffer_ptr ib(ggml_backend_alloc_ctx_tensors(single.get(), backend));
                GGML_ASSERT(ib);
                std::vector<int32_t> row_idx(n_sel*streams);
                for (int seq = 0; seq < streams; ++seq) {
                    memcpy(row_idx.data() + seq*n_sel, selected.data() + (seq*width + row)*n_sel, n_sel*sizeof(int32_t));
                }
                ggml_backend_tensor_set(idx, row_idx.data(), 0, ggml_nbytes(idx));
                auto * out = ggml_flash_attn_sparse(single.get(), q1, k, v, idx, 1.0f/16.0f);
                ggml_graph_clear(dec);
                ggml_build_forward_expand(dec, out);
                ggml_gallocr_ptr alloc(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
                GGML_ASSERT(ggml_gallocr_alloc_graph(alloc.get(), dec));
                GGML_ASSERT(ggml_backend_graph_compute(backend, dec) == GGML_STATUS_SUCCESS);
                std::vector<float> result(ggml_nelements(out));
                ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
                for (int seq = 0; seq < streams; ++seq) {
                    GGML_ASSERT(memcmp(result.data() + seq*24*256, b.data() + (seq*width + row)*24*256, 24*256*sizeof(float)) == 0);
                }
            }
        }
    }
    printf("kv=%d width=%d streams=%d strided=%d: mask union exact, attention max_abs=%.9g nmse=%.9g, decode/verify=%s\n",
        kv, width, streams, strided, max_abs, max_nmse, !bench && width <= 8 ? "exact" : "not checked");
    if (crossover) {
        measure_crossover(backend, dense, sparse, kv, width, permuted);
    } else if (bench) {
        for (auto * branch : {&dense, &sparse}) {
            auto * sel = ggml_new_graph_custom(branch->ctx.get(), 64, false);
            ggml_build_forward_expand(sel, branch->selection);
            auto * attn = ggml_new_graph_custom(branch->ctx.get(), 64, false);
            ggml_graph_add_node(attn, branch->out);
            printf("%s kv=%d width=%d: select_us=%.3f attention_us=%.3f qsa_graph_us=%.3f compute_buffer=%zu\n",
                branch == &dense ? "dense" : "sparse", kv, width, timed(backend, sel), timed(backend, attn),
                timed(backend, branch->graph), ggml_gallocr_get_buffer_size(branch->alloc.get(), 0));
        }
    }
}

static void measure_reserve(ggml_backend_t backend) {
    auto inputs = context();
    auto * q = ggml_new_tensor_4d(inputs.get(), GGML_TYPE_F32, 256, 1024, 24, 1);
    auto * k = ggml_new_tensor_4d(inputs.get(), GGML_TYPE_Q8_0, 256, 262144, 2, 1);
    auto * v = ggml_dup_tensor(inputs.get(), k);
    auto * top = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_I32, 512, 1024);
    auto * score = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_F32, 65536, 1024);
    auto * pools = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_I32, 4, 65536);
    auto * tail = ggml_new_tensor_2d(inputs.get(), GGML_TYPE_I32, 3, 1024);
    for (bool sparse : {false, true}) {
        qsa_graph branch(backend, sparse, q, k, v, top, score, pools, tail);
        size_t size = 0;
        ggml_gallocr_reserve_n_size(branch.alloc.get(), branch.graph, nullptr, nullptr, &size);
        bool dense_mask = false;
        for (int i = 0; i < ggml_graph_n_nodes(branch.graph); ++i) {
            dense_mask |= ggml_graph_node(branch.graph, i)->op == GGML_OP_TOP_K_MASK;
        }
        GGML_ASSERT(dense_mask != sparse);
        printf("reserve-only %s 262144x1024: allocator_bytes=%zu selection_bytes=%zu (includes unallocated input leaves)\n",
            sparse ? "sparse" : "dense", size, ggml_nbytes(branch.selection));
    }
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    const char * backend_name = "CPU";
    bool bench = false;
    bool crossover = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--bench") == 0) { bench = true; }
        else if (strcmp(argv[i], "--crossover") == 0) { crossover = true; }
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) { backend_name = argv[++i]; }
        else { fprintf(stderr, "usage: %s [--backend CPU|ROCm0|ROCm1] [--bench|--crossover]\n", argv[0]); return 1; }
    }
    if (bench && crossover) { fprintf(stderr, "choose either --bench or --crossover\n"); return 1; }
    ggml_backend_ptr backend(ggml_backend_init_by_name(backend_name, nullptr));
    GGML_ASSERT(backend);
    if (ggml_backend_is_cpu(backend.get())) {
        ggml_backend_cpu_set_n_threads(backend.get(), 2);
        ggml_backend_cpu_set_use_ref(backend.get(), true);
    }
    printf("backend=%s device=%s\n", ggml_backend_name(backend.get()), ggml_backend_dev_description(ggml_backend_get_device(backend.get())));
    if (crossover) {
        const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (strcmp(ggml_backend_reg_name(reg), "ROCm") != 0) {
            fprintf(stderr, "crossover calibration requires the ROCm backend\n");
            return 1;
        }
        printf("Crossover: supplied indexer scores -> TOP_K -> mask/compact selection -> Q8_0 attention; 21 alternating samples, five synchronous calls/sample.\n");
        for (int kv : {8192, 12288, 16384, 24576, 32768, 49152}) {
            for (int width : {1, 2, 3, 4, 5, 6, 7, 8, 32, 512, 1024}) {
                for (bool permuted : {false, true}) {
                    run_case(backend.get(), kv, width, 512, 1, false, true, true, permuted);
                }
            }
        }
    } else if (bench) {
        for (int kv : {4096, 32768, 131072, 235520}) {
            for (int width : {1, 2, 3, 4, 5, 6, 7, 8, 512, 1024}) {
                run_case(backend.get(), kv, width, 512, 1, false, true);
            }
        }
    } else {
        for (int width : {1, 2, 3, 4, 5, 6, 7, 8}) {
            run_case(backend.get(), 16384, width, 512, 1, false, false);
        }
        run_case(backend.get(), 16384, 8, 512, 2, true, false);
        run_case(backend.get(), 65536, 32, 512, 1, true, false);
    }
    measure_reserve(backend.get());
    return 0;
}
