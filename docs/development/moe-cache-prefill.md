# Direct MoE cache access during HIP prefill

This is the first implementation stage of the MoE cache work. The branch starts at `82478efb935b4e6a99946132553435d6f55fe8cb`: current `codex-compat` plus the existing, default-off measurement instrumentation. Do not merge it into production before checking both GPUs.

## Scope

The measurements of 2026-10-29 showed 146.76 GiB of expert H2D uploads and 4.24 GiB of cache-to-workspace copies over five large prefill batches. The median layer selected 344 of 512 experts. In that configuration, the existing shared cache groups of 44 of 48 layers already have at least 512 slots. These groups can hold any one layer's selection without a separate staging area.

This stage reuses those banks and the existing `MUL_MAT_ID` operation. It does not implement bounded 8/16/32-slot miss streaming. Groups whose worst possible selection does not fit retain the original temporary-weight path. It also does not add a hotness score, change cache budgets, or change speculative acceptance or KV state.

## Data flow

1. The model builds the same router results and selected expert IDs.
2. Graph construction checks whether `min(tokens * experts_used, experts)` fits the group's existing slots, without overflowing the product.
3. For more than 32 tokens, direct access requires an RDNA3 HIP device with wave32 and a bank of at most 32768 slots. Other backends retain their previous dispatch.
4. The existing slot-map copy callback reads the selected IDs and synchronizes the backend, as before. It pins the complete deduplicated selection while planning admissions.
5. Hits stay in their existing bank slots. Misses upload directly into their final slots. Gate/up/down read the same slot selection using the existing MMVQ/MMQ/MMB arithmetic.
6. The next callback synchronizes the same backend before it can reuse these slots. The callback cannot overwrite a slot while the preceding layer still reads it.

The prefill slot-map tensor is a host view of the existing map. It tells the copy callback to use prefill admission policy; it does not allocate a second map payload or a weight bank. The graph and its reservation are fixed by shapes and cache capacity, not by the runtime number of misses.

Prefill admissions stay in SLRU probation. Prefill hits refresh probation recency but do not promote entries into the protected segment or refresh protected recency. Decode/verify up to 32 tokens keep their previous policy. The existing victim order remains probation first, then protected if the current selection cannot otherwise fit; this is not an absolute prohibition on protected eviction.

Direct-cache matmuls retain the device ownership of their resident banks and slot lookup. Temporary-weight fallback matmuls retain stock scheduler placement. Forcing these full layouts onto GPU1 could increase its compute reservation beyond the roughly 398 MiB free in the supplied run. Moving these remaining layers requires bounded staging or a separate measured placement change.

MMB's descriptor scan now processes bank slots in blocks of 1024, carrying prefix totals between blocks. This preserves MMB for shared banks larger than one layer. Expert indices fit the descriptor's 16-bit field, and the 32768-slot bound also keeps the descriptor launch grids bounded. There is no change to dequantization, GEMM accumulation, GLU, routing rank or output reduction.

## Memory and performance expectations

Direct layers no longer request full-layout temporary expert copies through `sched_copy_experts()`. They still need slot indices, activation buffers and backend kernel scratch. Cache weight-bank allocation and the configured MoE cache budgets are unchanged.

Small groups still reserve full temporary layouts. Graph reservation includes the worst output shape, so removing many copies does not guarantee that the maximum compute allocation shrinks by their sum. Packed attention-mask reservation remains unchanged. Report the measured compute-buffer delta and process/driver VRAM separately.

Large banks increase routing-helper work and the number of empty MMB grid entries. Assigning prefill to the 7800 XT also changes traffic over its x4 link. The CPU tests cannot establish a throughput gain or rule out a GPU regression. Those are explicit hardware-test gates before merge.

## Targeted validation

Use the same ROCm 10.1 environment and HIP configure arguments as the working production build, including both `gfx1100;gfx1101` targets. Build `test-llama-archs`, `test-backend-ops` and `llama-server`. The existing `LLAMA_MOE_MEASURE` build option enables tracing; it remains off by default and is not needed for execution.

Policy invariants, including repeated IDs, selection pinning, cross-layer eviction and overflow-safe capacity decisions:

```sh
./build-rocm-moe-prefill/bin/test-llama-archs --test-moe-cache-policy -s 42
```

Large-bank numerical comparisons against the CPU reference, for Q8_0 and IQ3_S, broadcast and down-projection inputs, and GLU. The six cases include 1024, 1025 and 3202 slots, unsorted IDs, hot experts and sparse tail slots:

```sh
./build-rocm-moe-prefill/bin/test-backend-ops test -o MOE_CACHE_BANK -b ROCm0
./build-rocm-moe-prefill/bin/test-backend-ops test -o MOE_CACHE_BANK -b ROCm1
```

The Qwen4Exp model fixture uses quantized Q8_0 expert banks and Q8_0 KV. It compares all logits against the same model without a cache for widths 64/512/1024 and 1/4/8, including state restore and warm-cache sequence reset. Equal, insufficient and zero budgets exercise direct access and fallback. Its callback verifies device ownership of direct-cache matmuls. Run the combined case to exercise unequal per-device budgets as well:

```sh
./build-rocm-moe-prefill/bin/test-llama-archs --test-moe-cache-prefill -s 42 -b ROCm0
./build-rocm-moe-prefill/bin/test-llama-archs --test-moe-cache-prefill -s 42 -b ROCm1
./build-rocm-moe-prefill/bin/test-llama-archs --test-moe-cache-prefill -s 42
```

Without an RDNA3 HIP device, this command validates the quantized model fixture on CPU and explicitly reports that GPU cache execution was skipped. A CPU pass is not a test of the HIP descriptor kernel or cache callback execution.

The fixture accepts logit NMSE below `5e-4` and prints the worst value. This numerical tolerance is not a measured acceptance bound for the production IQ3_S model. Confirm DFlash acceptance, Vision, prompt-tail ingest, checkpoints/rollback and SSD prompt-cache restore on that model.

## Collect the first hardware comparison

Keep the current `15200,9600` MiB cache budgets, tensor split, context, batch size and malloc-stage-pool settings. Compare the baseline `82478ef` with this branch, using the existing [measurement tools](moe-measurement.md). Run the same request once with a fresh process and then repeat it in that process with prompt caching disabled, so the second request measures a warm expert cache and a fresh prefill.

For this comparison, keep SSD and RAM prompt caches disabled. Do not clear the user's on-disk cache. Keep identical sampling settings and record response hashes and DFlash counts; a matching seed alone does not establish an identical generated workload. The instrumented and uninstrumented builds must use the same source revision and HIP flags.

Check these results before expanding the matrix:

- Cached prefill produces `admission_h2d` and `bank` records instead of `work_copy`/`d2d` for eligible layers.
- GPU1 owns its direct-cache prefill layers; residual temporary copies retain stock placement.
- Warm-request expert H2D bytes decrease; report cold and warm results separately.
- Prefill and generation throughput do not regress, and DFlash acceptance remains comparable on the same workload.
- Compute reservations, profiler allocation peaks and driver VRAM are recorded independently.

Hardware results and the resulting throughput/VRAM deltas are pending. Bounded miss streaming and adaptive admission for the remaining small groups are subsequent work after this stage has a measured baseline.

## Validation in the implementation environment

- CPU Release build with measurement enabled and disabled.
- SLRU policy test: capacity checks, duplicate IDs, selected-slot retention, probation/protected invariants, generation-counter wrap and 2000 mixed plans.
- Six `MOE_CACHE_BANK` numerical reference cases passed on CPU, including Q8_0/IQ3_S and GLU. This does not execute the HIP MMB scan.
- Quantized Qwen4Exp CPU fixture: all tested logits matched exactly (NMSE 0), including checkpoint restore and sequence reset. GPU cache execution is explicitly skipped here.
- Existing QSA CPU oracle: 12 attention comparisons each for F16 and Q8_0 KV, with exact logits after sequence reset and checkpoint rollback.
- Existing measurement analyzer tests: 5 passed.
- HIP configuration could not complete: no HIP compiler/ROCm root is installed. The HIP build, both GPU test sets and production-model performance/compatibility checks remain open.
