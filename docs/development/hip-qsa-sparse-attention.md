# Native HIP QSA sparse attention

Base: `codex-compat` at `0c176bf35b19a56b67a465bd293c108af58b9fe3`.
Branch: `feature/hip-qsa-sparse-attention`.

This change only addresses Qwen4Exp / Qwen3.8 Flash-Next QSA. It does not change the MoE cache or prefill GEMM, allocator policy, model quantization, RoPE, persistent KV format, DFlash inputs, or prompt-cache format. There are no new inference switches or environment variables.

## Data flow

The existing lightning indexer, pool visibility scores and unordered Top-K ranking remain unchanged. On eligible layers, `build_qsa_sel()` returns `QSA_SELECT(top_k, score, pool_idxs, tail_idxs, n_kv)` instead of building `TOP_K_MASK` or its scatter expansion.

`QSA_SELECT` reads only the picked pool scores and their members, plus the incomplete tail. Pools with `!(score > -INF)` are excluded, as in `TOP_K_MASK`. Negative, sentinel and out-of-range cells are discarded. Cells are sorted, deduplicated and packed into an I32 list per query, with `-1` padding. Sorting cell addresses does not change Top-K ranking. Sequence membership and causal visibility come from the unchanged per-query pool scores and tail inputs; no position-based approximation is introduced. The HIP implementation uses a shared-memory bitonic sort of at most 4096 entries and a block prefix scan, without scanning KV cells or a dense mask.

`build_attn_qsa()` stores K/V and applies/undoes the existing cache rotations as before. For compact selections it creates `FLASH_ATTN_SPARSE(Q, K, V, indices)`. All four operands are ordinary graph dependencies. The scheduler transfers the existing small host inputs and keeps the produced indices on the GPU; no host readback, index scan, or synchronization is added.

The two distinct GGML ops prevent another backend from accepting Flash Attention while silently ignoring sparse operands. CPU implementations provide reference evaluation. Other architectures and backends keep their existing model graph.

## HIP attention and automatic dispatch

The HIP implementation is in `ggml/src/ggml-hip/fattn-qsa.cu`. It uses the existing RDNA tile arithmetic through a compile-time indexed specialization of the native Q8_0 tile loader and attention iteration. The dense specialization is unchanged. Only selected K/V rows are fetched and dequantized into shared tiles; there is no whole-cache F16 conversion, gathered K/V copy, dense attention mask, or mask-scanning prepass. The NVIDIA sparse MMA implementation is untouched.

Supported configurations:

- gfx1100 and gfx1101, with Flash Attention enabled and KQV offloading enabled.
- F32 queries, Q8_0 K and V, 256-element K/V heads, GQA ratio divisible by four.
- Per-query selections with up to 4096 slots; query widths up to 1024 per stream.
- Strided cache views and multiple KV streams, with matching query/selection stream layouts.
- The actual layer device must support both ops and own compatible K/V buffers. This also works with the dummy buffers used for allocation-only model loading.

Each block handles one query and four query heads, with four Wave32 waves and 32 selected cells per iteration. Decode and verify widths 1-8 use exactly the same tile configuration and 16 KV partitions. Prefill uses the same indexed iteration with one partition per query/head group. Softmax maxima/sums, QK accumulators and the final partition combination use F32; Q conversion, probabilities and VKQ accumulation retain the existing tile kernel's FP16 arithmetic. This is not a claim of bitwise equivalence to the dense GPU kernel: compacting cells changes KV partition membership and accumulation order. Empty/invalid-only lists produce zero outputs.

The model graph uses a conservative automatic crossover:

```
n_kv >= max(8192, n_sel * (queries_per_stream <= 8 ? 4 : 16))
```

For the usual 2051-slot selection this keeps short contexts dense and requires more sparsity before replacing a prefill WMMA path. These ratios are engineering defaults, not measured crossover points. The explicit GGML op remains available to the benchmark helper at short contexts so that both kernels can be compared without an inference flag.

F16/BF16/Q4 or mixed K/V, different head sizes/GQA, other GPUs/backends, Meta devices, incompatible cache placement, large unsupported strides, ALiBi, softcap, training, and shapes outside the bounds retain the old QSA mask path. There is no CPU fallback selected intentionally for production sparse attention.

## Allocation and reservation

For an eligible QSA layer the graph no longer contains:

- The full-width QSA selection mask, or its scatter/dump-row construction.
- The broadcast zero/keep vector used by the fused mask producer.
- Full-width QSA mask scans and their GPU scratch.

The ordinary KQ-mask tensor may still exist as graph metadata. With no consumer it has no buffer and its existing input-fill guard skips it. Dense layers or fallback QSA layers can still consume their masks normally.

The global `kq_mask_packed_reachable()` / `graph_reserve(..., packed_kq_mask)` protection is unchanged. It still reserves packed masks for consumers that need them, including non-derivable multimodal cases. QSA uses the same capability/crossover decision during reservation and execution; its transient indices are counted instead of a QSA mask. No global reserve margin was removed.

Decode/verify partition results and F32 metadata are appended to the sparse attention node's allocation. `get_alloc_size` and `ggml_op_alloc_size_may_expand()` include them before scheduling; no private unaccounted allocation or new cache is used. Prefill needs no partition buffer. Existing indexer scores, pool masks, Top-K tensors and persistent K/V storage remain.

A reserve-only CPU GGML allocator experiment for one QSA subgraph, with `n_kv=262144`, 1024 queries, 24 Q heads, two KV heads and 65536 indexer pools, measured:

| Quantity | Existing mask graph | Compact graph |
| --- | ---: | ---: |
| Selection tensor | 536870912 bytes (512 MiB) | 8400896 bytes (8.0117 MiB) |
| Measured allocator requirement | 1119367168 bytes | 590372864 bytes |

The measured difference is 528994304 bytes (504.4883 MiB), including the removed 0.5 MiB keep vector. The fixture includes unallocated input leaves in both measurements; it is not the full llama scheduler or a GPU peak measurement. Real compute-buffer savings depend on lifetimes, other layers, device splitting and the remaining indexer tensors. The test asserts that the compact graph has no `TOP_K_MASK` node. No 512 MiB end-to-end VRAM saving is claimed.

## Validation completed

- CPU Release build: llama/common and the three targeted test binaries succeeded.
- `test-backend-ops`: 21/21 CPU cases for `QSA_SELECT` and `FLASH_ATTN_SPARSE`, including real D=256 / GQA=12 Q8_0 shapes, widths 1-8/32, cache views, unsorted indices, invalid entries, empty selections and multiple streams. CPU support CSV records these actual results.
- `test-qsa-sparse`: ten dense-mask-versus-compact comparisons with four input/visibility updates each. Unordered/repeated picks, overlapping/duplicate members, incomplete/padded pools, sentinels, per-query selection, strided inputs and two streams are covered. Selected cell sets are exact. CPU attention maximum absolute error and NMSE are both zero. For widths 1-8, every verify row is also recomputed as decode and compared byte-for-byte.
- `test-qwen4exp-qsa-equivalence`: the existing small Qwen4Exp fixture evaluates the compact CPU selection/attention at live QSA boundaries and feeds the result into subsequent model nodes. F16 and Q8_0 KV each pass 12 attention comparisons and byte-identical full-model logits, including a second sequence reset and restoration of a checkpoint containing recurrent/KV/indexer state. Production CPU dispatch remains dense; this is a test-only numerical oracle.
- Nine targeted CTests passed: the two new oracles plus graph recovery, HC layer inputs, DFlash device inputs, vision, both tail-ingest modes and derived-mask equivalence.
- `git diff --check` passed.

HIP configuration was attempted with `gfx1100;gfx1101` and stopped at `Failed to find HIP root directory`. This environment has no ROCm SDK, `/dev/kfd`, AMD GPU, Vulkan SDK/device, or production Swift/DFlash GGUFs. Consequently the HIP code has **not** been compiled here. GPU numeric results, GPU decode/verify equality, full-model GPU logits, acceptance, performance, crossover tuning and actual peak VRAM remain open. Vulkan execution/build is also untested; its model path and backend implementation are unchanged.

## Hardware review commands

Build the targeted binaries with the existing ROCm configuration:

```sh
cmake --build build-rocm10 --target test-qsa-sparse test-backend-ops test-llama-archs
build-rocm10/bin/test-qsa-sparse --backend ROCm0
build-rocm10/bin/test-qsa-sparse --backend ROCm1
build-rocm10/bin/test-backend-ops test -b ROCm0 -o 'QSA_SELECT,FLASH_ATTN_SPARSE'
build-rocm10/bin/test-backend-ops test -b ROCm1 -o 'QSA_SELECT,FLASH_ATTN_SPARSE'
```

The oracle compares dense and compact outputs on the same backend, prints maximum absolute error/NMSE, and checks exact sparse decode/verify consistency. Its GPU numerical tolerance is the existing FA backend-test NMSE bound, `5e-4`; this alone does not establish stable DFlash acceptance. The normal backend tests also compare HIP against the CPU reference.

For the requested synthetic matrix:

```sh
build-rocm10/bin/test-qsa-sparse --backend ROCm0 --bench
build-rocm10/bin/test-qsa-sparse --backend ROCm1 --bench
```

This covers KV lengths 4096, 32768, 131072 and 235520; widths 1-8, 512 and 1024. For each branch it reports selection/mask preparation, attention-only and combined QSA graph wall times, and that branch's allocated compute buffer. Indexer scoring and Top-K ranking are supplied as inputs and are excluded from these timings. The helper holds both references simultaneously, so process-level VRAM from this helper is not an inference peak measurement. A separate `test-backend-ops perf` filter for the two new ops provides the same shape matrix.

Before merge, compare the base and candidate in separate fresh processes on the production model, using identical prompts, batches/ubatches, GPU split and fixed MoE-cache size. Record end-to-end prefill and generated tokens/s, accepted/proposed DFlash tokens, logits/error statistics, actual scheduler compute reservations, peak device memory, and any mid-run reallocation. Include long text, multimodal inputs, prompt-tail ingest, sequence edits, checkpoint/SSD-cache restore and rollback. The short-context crossover needs measured confirmation on both cards. These results must be added before asserting a performance or acceptance improvement.

## Selection follow-up

The hardware-tested e7b431d remains the inference baseline. The [selection candidate comparison](hip-qsa-select-candidates.md) documents the next test-only optimization round; candidate promotion and crossover calibration await measurements on both GPUs.
