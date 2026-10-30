# Window-based MoE admission and bounded HIP expert streaming

Review branch: `feature/moe-cache-prefill`. Production `codex-compat` is not changed.
The branch includes the earlier, default-off `LLAMA_MOE_MEASURE` build instrumentation.
No new application option, runtime environment variable or cache budget is introduced.

## Why replace the first experiment

The hardware comparison of `82478ef` and `085c722` showed that removing expert D2D
copies alone was insufficient: warm prefill fell from 448.03 to 318.72 tokens/s,
although D2D bytes fell by about 97% and GPU0 compute reservation fell by 500.72 MiB.
The shortcut passed physical cache slot IDs to ordinary `MUL_MAT_ID`. A shared bank
of 3202 slots then became a 3202-expert execution grid for a 512-expert model, also
changing tile selection. Admitting every prefill miss caused cross-layer churn.
These measurements describe the **old experiment**, not this implementation.

## Execution and ownership

The router, expert ranking, FFN activation, LoRA, Hadamard transform, weight scales
and output reduction retain their existing semantics. One host planner observes
all supported cached requests. The configured resident banks and their capacities
remain in place; SLRU admission/eviction is replaced by window-based demand.

The existing slot-map copy boundary still reads the selected IDs and synchronizes
that backend once per layer. It deduplicates the selection and plans three classes:

| Class | Weight location | Work-weight copy |
| --- | --- | --- |
| Resident hit | Existing cache slot | None |
| Admitted miss | Upload directly to its final cache slot | No staging-to-cache bounce |
| Bypassed miss | At most 32 experts in shared device scratch | Reused after its readers |

Every resident selected by the layer is pinned through all its gate/up/down
operations. Admission fills all weight banks of the expert bundle. The next
slot-map callback synchronizes the owning backend before it may evict those slots.
Admission ranking and physical placement are separate: ascending admitted experts
are placed in ascending freed slots to retain coalesced H2D copies. Consecutive
bypassed experts are uploaded together too.

For more than 32 query tokens, the process-local `MUL_MAT_ID_STREAM` operation
consumes **logical model IDs**, a lookup dependency and a backend packet callback.
Each packet supplies a logical-ID-to-weight-address table, a compact active list,
its routed row count and maximum rows per expert. Packet zero covers resident and
newly admitted experts. Remaining packets cover at most 32 bypassed experts each.
Pointers and active lists are GPU-resident during execution. Their host upload
sources are distinct for every bank/packet and survive until the next routing
synchronization, including when the optional malloc-stage-pool filter excludes
metadata.

MMQ builds its logical routing table and quantizes activations once per matrix
operation, then launches only active experts, using their real weight addresses.
Tile selection uses the model's expert count, never the resident bank capacity.
MMB similarly converts activations and builds routing once; per-packet descriptors
contain logical expert IDs and use the packet's routed rows, not all cache slots.
Neither kernel changes weight dequantization or dot-product arithmetic. Separate
compile-time indirection specializations preserve the legacy MMQ/MMB kernels.
The mixed IQ3_S model's IQ3_XXS, IQ2_S, Q2_0, IQ4_NL and IQ4_XS banks use existing
native MMQ formats; MMB remains selected by its existing shape/type predicate.

Gate/up/down are ordinary separate graph operations, or the existing merged
`gate_up` weight operation. A staged weight matrix is uploaded once when that
operation consumes it. Its storage can be reused by the next packet/matrix on the
same stream. This implementation does not add a streaming-specific gate+up/GLU
fusion or split one FFN into a separate scheduling subsystem.

The GGML graph and reservation remain fixed for a given shape. Runtime packet
counts do not add graph nodes or reserve full expert working tensors. These splits
execute outside HIP graph capture because their H2D uploads and launch counts vary.
Small all-resident decode/verify retains its existing kernels and graph eligibility.

## Demand window and small batches

Each `(layer, expert)` counts repeated requests for its weights, rather than raw
token assignments. Sixteen capped counters cover 4096 advancing token positions
in 256-token slices. The newest four slices receive four times the older weight.
Token assignment frequency is only a tie-breaker, after demand and recency.
A miss must exceed the weakest unpinned resident's score by more than
`max(1, resident_score/8)` to replace it; an empty slot accepts immediately.

The clock follows sequence position progress, independently per sequence. Revisiting
positions during verify or rollback does not advance it repeatedly; the same
progress stamp cannot increase an expert's demand counter again. A prompt at position zero after previous forward progress begins another segment. This is a conservative progress proxy,
**not** an accepted-token oracle: a first speculative visit can count demand and
advance the high-water mark by its lookahead. Policy state is performance metadata
and is not serialized with KV checkpoints. Restoring a nonzero prefix does not
rewind this clock; demand may therefore adapt more slowly until progress resumes.
No speculative result, token acceptance decision or saved KV content is changed.

For up to 32 tokens, if the worst-case selection fits, the same policy chooses
victims but guarantees residency of the current selection. This avoids inserting
packet uploads or disabling graph capture on the latency-sensitive MMVQ path.
Admission hysteresis/bypass applies to the streaming plan; these small requests
can force admission. An all-hit plan returns before building a victim heap. Miss
plans use a minimum heap rather than sorting all cache slots. Small groups that
cannot guarantee residency use the existing host-weight fallback.

## Memory, fallback and limits

The additional persistent device scratch is bounded by:

```
33 * largest_single_weight_matrix_bytes_per_expert_on_device
+ max_logical_experts_on_device * (sizeof(uint64_t) + sizeof(int32_t))
+ backend alignment
```

There is one 32-slot scratch bank plus one read-padding guard per device, shared
across layers and gate/up/down banks. It is not 32 complete expert bundles per
layer. Resident bank budgets are unchanged. Both allocations are included in
`llama_context::memory_breakdown()` and reported separately at initialization.
The scratch is outside the configured resident-cache budget; increasing the
resident cache should wait for measured peak VRAM on the real model.

Supported streamed operations eliminate their full-layout temporary expert-weight
inputs and cache-hit D2D copies through `sched_copy_experts()`. They retain
activations, routing tables, and MMQ/MMB scratch. This is an allocation/dataflow
property; a whole-process VRAM delta cannot be inferred by summing logical tensor
sizes. Existing conservative attention-mask reservations are untouched.

Initial automatic streaming is restricted to RDNA3.0 wave32 HIP, contiguous native
MMQ quantized weights, K aligned to their native quantization block, M divisible
by 128, at most 32768 logical
experts and fewer than 65536 routed rows. MMB retains its existing requirements.
Unsupported types (including separate-scale FP4), activation layouts/precisions,
backends and zero-budget groups use the existing host-weight path. The operation
is never sent to RPC: the client and server reject its process-local callbacks.
CPU supports a scalar test oracle; model construction enables streaming only on
supported HIP devices. Vulkan uses the existing operations.

Work stays on the bank's owning GPU. This does not implement cross-device routing
of misses, peer weight reads or separate transfer/compute-stream overlap. In
particular, the 7800 XT still has to receive its missing weights over its x4 link.
The SSD prompt cache, mmap malloc-stage-pool workaround, DFlash device inputs,
Vision, prompt-tail ingest and checkpoint/rollback mechanisms are not modified.

## Validation and current evidence

Implementation environment: CPU Release builds with measurement enabled and disabled;
the latter also builds the RPC backend. All three targeted CTest entries and six
measurement-analysis tests pass. No HIP compiler or GPU is present;
HIP configure fails with `Failed to find HIP root directory`. Consequently the HIP
build, kernel numerics and end-to-end speed/VRAM results below remain hardware gates.

- Window-policy tests: 2000 mixed plans, duplicate routes, pinned hits, cross-layer
  eviction, forced admission, bypass beyond capacity, hysteresis, retry suppression,
  sequence progress/reset, expiration and 4096 scalar-vs-packed score comparisons.
- `test-moe-stream`: 27 CPU numerical cases across Q8_0 and all six expert formats
  observed in the supplied mixed model. Noncontiguous/reversed resident slots,
  unsorted logical routes, 8/16/32-slot packets, incomplete tails, broadcast/up and
  down inputs, and an empty resident packet/all-temporary selection are covered. Six cases use
  K=640 for Q8_0, IQ4_NL and Q2_0, the last two matching the real down-bank formats.
  NMSE is at most 7.2e-15; maximum absolute difference is 3.06e-5. Activations are
  exactly representable in Q8 and BF16 to isolate routing/address correctness.
- Quantized Qwen4Exp CPU fixture with expert FFN width 640: exact logits (NMSE 0) for prefill 64/512/1024,
  decode/verify 1/4/8, checkpoint restore and warm-cache sequence reset. It explicitly
  skips GPU-cache execution when no supported device exists.
- Existing measurement analysis tests include staging H2D bytes; host enqueue times
  remain distinct from GPU durations.

A sanity replay of the supplied `085c722` measurement trace with unchanged group
capacities/bundle sizes predicts 71.94 GiB cold and 65.26 GiB warm prefill expert
uploads, compared with the trace's 139.66/103.59 GiB. This replay uses recorded route
histograms and a graph-width token-clock proxy; positions and accepted/rejected
verify histories were not recorded. It models neither DMA/compute overlap nor
kernel, planning or graph costs. It is an admission-policy estimate, not a measured
bandwidth, throughput or acceptance result, and is not the perfect-future oracle.

## One focused hardware round

Use the normal ROCm 10.1 build configuration with both `gfx1100;gfx1101` targets.
Build `llama-server`, `test-moe-stream` and `test-llama-archs`.

```sh
./build-rocm/bin/test-moe-stream --backend ROCm0
./build-rocm/bin/test-moe-stream --backend ROCm1
./build-rocm/bin/test-llama-archs --test-moe-cache-policy -s 42
./build-rocm/bin/test-llama-archs --test-moe-cache-prefill -s 42
```

The kernel fixture compares streamed results with the ordinary operation **on the
same GPU**, before testing integrated model/cache behavior. It exercises MMQ for
64 tokens and MMB where applicable for 512 tokens. The model fixture checks
single/unequal/zero budgets, device ownership, prefill and decode/verify, state
restore and sequence reset. Its printed logit tolerance is NMSE <5e-4; that is not
an acceptance guarantee for the production model.

After correctness passes, compare production `82478ef` and this branch with the
existing [measurement suite](moe-measurement.md), holding `15200,9600` MiB cache
budgets, tensor split, malloc pool, context and batch configuration constant.
`085c722` is the known regressing intermediate experiment, not the performance
baseline. Keep SSD/RAM prompt reuse disabled for the controlled cold/warm prefill
comparison, without deleting the on-disk prompt cache. Test SSD restore normally
in the production-model compatibility check.

Report uninstrumented throughput first, then measured H2D/D2D bytes and routing /
planning host times, and profiler kernel durations, allocation lifetimes and peak
VRAM separately. Measurement builds emit `hybrid_plan`, `admission_h2d` and
`stage_h2d`; the analyzer counts both H2D classes. Verify DFlash acceptance and
Vision/tail/checkpoint behavior on the real model. Do not infer a speedup from the
isolated kernels or a lower byte count. Do not merge to `codex-compat` before that
review.


## 2026-10-30: correct the inactive K=640 dispatch

The two `moe-stream-ab-20261030-074022-834702` and
`moe-stream-ab-20261030-074408-198202` handoffs do **not** benchmark active expert
streaming. Their server build IDs are `82478ef` and `a4aaa9b` as intended, and the
candidate emits the new `hybrid_plan` records. However, it reserves no bounded miss
scratch, emits no `stage_h2d`, and still emits 705 `work_copy` records per request.
Both profiled device allocation peaks are identical between baseline and candidate.

The cause is an overly restrictive implementation guard: the whole cache group
required K to be divisible by 256, copying MMB's limit into streaming eligibility.
The actual model has `n_embd=2560` and expert FFN width 640. Every down projection
therefore has K=640, excluding the complete group even though its IQ4_NL/Q2_0
format is valid for the existing MMQ kernel. MMB's inability to execute that one
matrix must not disable MMQ streaming for the layer.

The corrected model and HIP guards share `ggml_moe_stream_weight_supported()`.
They require the format's native block alignment; MMB keeps K % 256 and declines
K=640 independently, routing those operations to MMQ. No dot-product arithmetic,
router policy, cache size or runtime switch is changed by this correction. The
model fixture now uses FFN width 640 rather than 512, including Q8_0 gate/up and
down weights. Kernel fixtures cover K=640 at 64/512 queries, mixed resident and
staged weights and an all-temporary plan. They assert the same shared eligibility
predicate used by model construction and HIP supports-op.

For example, the first handoff's large-prefill-only transfer counts are:

| Metric | Baseline | Inactive candidate |
| --- | ---: | ---: |
| Cold expert H2D | 146.7583 GiB | 146.7583 GiB |
| Warm expert H2D | 104.5918 GiB | 104.5996 GiB |
| Warm cache-to-workspace D2D | 46.3959 GiB | 46.3959 GiB |

These figures exclude decode/verify. The runner's overview combines whole-request
host bytes with uninstrumented timings, so those columns must not be mistaken for
prefill-only transfer measurements. The first baseline's cold prefill (223.53 t/s)
was much slower than the same baseline binary in the second session (348.61 t/s);
its apparent +55.82% candidate gain is not evidence for this inactive streaming
path. In the second session warm prefill differs by only +1.65%. Different cold
completion hashes and DFlash work prevent a clean generation comparison too.

The checked-in runner has automatic activation gates:

```sh
python3 scripts/moe-measure/stream-ab.py --profile
```

It uses the same four build directories and existing Fish templates/request. It
requires bounded scratch on every cache GPU **before** submitting candidate
requests, then requires `hybrid_plan`, no `work_copy` in either request and cold
`stage_h2d`. This is specific to the suite's Flash-Next setup; intentionally testing
a model/configuration that falls back requires a different workload, not treating
that fallback as a streaming result. The checks reject both submitted inactive
archives and accept a synthetic active trace; a partial GPU activation is rejected.
The original uploaded runner can still produce timings, but does not enforce these
gates. Rebuild the two candidate directories after pulling the fix. Run the two
GPU numerical fixtures and integrated model test above before the next A/B round;
the inactive runs establish neither new-kernel numerical correctness nor its
throughput or VRAM benefit.

Correction validation: CPU Release with measurement enabled; all three targeted CTest
entries pass (27 packet comparisons, policy invariants and the FFN-640 model fixture).
The six existing analyzer/profile unit tests pass. No HIP SDK or GPU is available
in the implementation environment, so the corrected GPU dispatch, GPU numerics
and resulting end-to-end performance/VRAM changes remain unverified here.
