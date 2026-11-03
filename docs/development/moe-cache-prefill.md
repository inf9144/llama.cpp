# Window-based MoE admission and bounded HIP expert streaming

Review branch: `feature/moe-cache-prefill`. Production `codex-compat` is not changed.
The `21ddbbac` hardware run validates redirected prefill and lower peak device
allocation on the supplied workload. The additional decode/verify and transfer
pipeline package below still needs numerical and performance validation on GPUs.
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

Large requests retain the slot-map copy boundary: it reads the selected IDs and
synchronizes that backend once per layer. Supported requests with at most eight
queries instead resolve slots on the GPU and notify a host admission worker.
Both use the same policy and three weight classes:

| Class | Weight location | Work-weight copy |
| --- | --- | --- |
| Resident hit | Existing cache slot | None |
| Admitted miss | Upload directly to its final cache slot | No staging-to-cache bounce |
| Bypassed miss | At most 32 experts in shared device scratch | Reused after its readers |

Every resident selected by the layer is pinned through all its gate/up/down
operations. Admission fills all weight banks of the expert bundle. A host
slot-map callback synchronizes the owning backend before eviction; a device miss
is published only after the preceding layer's readers have completed on its stream.
Admission ranking and physical placement are separate: ascending admitted experts
are placed in ascending freed slots to retain coalesced H2D copies. Consecutive
bypassed experts are uploaded together too.

Large-batch placement is fixed when the context is created. On Linux, supported
HIP devices expose a PCI bus ID; sysfs provides each link's speed ceiling and
negotiated width. Resolve the endpoint symlink and follow its PCI bridge parents;
the smallest link product on that path determines the estimate. For a partially resident group, compare the owner's estimated
uncached transfer cost with full host streaming on a faster link. Redirect only
when `faster_link * (total_experts - slots) > owner_link * total_experts`.
Unknown links, equal links, complete banks, and selections wider than 32 experts
retain local placement. This is a static transfer-cost estimate, not a measured
bandwidth or a guarantee for every router distribution or PCIe topology.
Speed parsing uses the C locale. If an endpoint or PCI parent has unreadable or
invalid link data, the whole path is unknown rather than skipping a possible
bottleneck. Startup reports the limiting PCI device and its speed/width once per
supported GPU, or explicitly reports unknown topology.

Redirected groups use the same streaming operation and a typed view of the
destination's bounded scratch. They do not copy a remote cache bank or upload
prefill admissions into it. All selected weights are streamed from host memory;
the remote bank remains available for small decode/verify. Demand is still
observed for admission on later local requests. Local groups retain direct cache
reads, direct admission and miss staging. Startup logs announce each owner's
prefill execution device. No peer access or remote weight cache is introduced,
and the graph cannot switch execution devices during a request.

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
operation consumes it. Two bounded lanes overlap the next upload with the current
packet's computation; ready/consumed events protect every lane's storage and metadata.
This implementation does not add a streaming-specific gate+up/GLU
fusion or split one FFN into a separate scheduling subsystem.

The GGML graph and reservation remain fixed for a given shape. Runtime packet
counts do not add graph nodes or reserve full expert working tensors. These splits
execute outside HIP graph capture because their H2D uploads and launch counts vary.
Small decode/verify retains graph eligibility, including the device lookup wait.

## Demand window and small batches

Each `(layer, expert)` counts repeated requests for its weights, rather than raw
token assignments. Two windows use sixteen capped counters, each covering 4096
advancing positions in 256-position slices. One sees all requests; the other
advances only with batches of at most 32 tokens. A long prefill therefore cannot
expire recent decode/verify demand. Each window weights its newest four slices
four times more strongly. Scores order latency-window demand first and total
demand second: `latency_score * 421 + total_score`, where 420 is the maximum
single-window score. There is no fixed partition of resident storage.
Token assignment frequency is only a tie-breaker, after demand and recency.
A miss must exceed the weakest unpinned resident's score by more than
`max(1, resident_score/8)` to replace it; an empty slot accepts immediately.

The clock follows sequence position progress, independently per sequence. Revisiting
positions during verify or rollback does not advance it repeatedly; the same
progress stamp cannot increase an expert's demand counter again. The two windows
have separate request stamps, so a first small request can record latency demand
even if prefill already visited that position. A prompt at position zero after previous forward progress begins another segment. This is a conservative progress proxy,
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
2 * 33 * largest_single_weight_matrix_bytes_per_expert_on_device
+ 2 * max_logical_experts_on_device * (sizeof(uint64_t) + sizeof(int32_t))
+ sum(cached_streaming_layers * logical_experts * sizeof(int32_t))
+ backend alignment
```

Each execution device has two 32-slot scratch banks with one read-padding guard each, shared
across layers and gate/up/down banks. They are not complete expert bundles per
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

Small-batch work stays on the bank's owning GPU; large-batch placement follows the
link estimate above. There are no cross-device cache reads. Small misses owned by
the 7800 XT still upload over its x4 link. A nonblocking upload stream overlaps
large packet transfers with computation, without changing the cache's capacity.
The SSD prompt cache, mmap malloc-stage-pool workaround, DFlash device inputs,
Vision, prompt-tail ingest and checkpoint/rollback mechanisms are not modified.

## Validation and current evidence

The initial implementation was validated in CPU Release with measurement enabled
and disabled, including the RPC backend, three targeted CTest entries and ten
measurement/runner tests. Subsequent sections record each revision's checks.
An official ROCm SDK is now available for compilation, but no GPU is available;
kernel numerics and end-to-end speed/VRAM remain hardware gates for new changes.

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
single/unequal/zero budgets, local/redirected placement, prefill and decode/verify, state
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
All ten analyzer/profile/runner unit tests pass, including preserving the handoff
when activation fails before candidate requests complete. No HIP SDK or GPU is available
in the implementation environment, so the corrected GPU dispatch, GPU numerics
and resulting end-to-end performance/VRAM changes remain unverified here.


## 2026-10-30: active streaming exposes a poisoned read guard

The `moe-stream-ab-20261030-085959-917988` handoff proves active streaming on both
GPUs: each reserves 29.01 MiB of bounded scratch, and the candidate emits
`stage_h2d` with no `work_copy` or expert D2D records. However, every candidate
request accepts **0 of 375 draft tokens** and produces repeated `必究` characters.
The baseline produces coherent text and accepts 78/144 or 83/130 draft tokens.
This is a correctness failure. Neither the roughly 900/1277 tokens/s cold/warm
prefill nor the 20.18/10.61 GiB whole-request H2D totals are a valid performance
result for the intended workload.

The first cold prefill graph has exactly matching baseline/candidate router
histograms through layer 20. Starting at layer 21, all 1024 queries select only
experts 0..9; later layers remain collapsed. For GPU1's large batches that means
only 10 distinct experts per layer instead of hundreds. Much of the apparent
transfer improvement therefore comes from corrupted routing, not useful reuse.
The fixed graph allocation still shows the intended main GPU0 compute reservation
change, 1176.75 -> 676.03 MiB, with 29.01 MiB extra staging on each device. A valid
end-to-end VRAM/speed conclusion must wait for correct production-model execution.

A concrete staging-padding bug explains how non-finite outputs can arise. MMQ
loads complete K tiles. K=640 leaves a final partial 256-value tile, so its weight
loads can reach beyond the last row of the last staged matrix. Null activation
padding does not neutralize a NaN/Inf weight scale: IEEE multiplication by zero
still produces NaN. The bank's extra expert was cleared at initialization, but
reusing the same raw storage for different matrix sizes/formats can put old
quantized payload into that guard. In particular, a 460800-byte Q2_0 down expert
can follow a 627200-byte IQ3_XXS up expert. Its new end lies within the old payload,
whose bit fields need not be valid finite Q2_0 scales.

The packet contract now exposes the staging read guard (one expert immediately
after the current payload). Both HIP consumers clear it with `cudaMemsetAsync`
on the executing backend stream, after the packet's uploads and before its
kernels. Resident cache packets do not clear or copy their weights. The existing
33-slot allocation already covers the guard, so no extra VRAM, host staging cache,
feature switch, or host synchronization is added. CPU's packet oracle implements
the same guard contract. Matmul arithmetic is unchanged.

Regression coverage now poisons the staging payload and guard with 0xff **before
every packet**, including the last short packet, then checks finite/equivalent
outputs and a zero guard. This models reuse across quantized matrices rather than
freshly cleared test storage. A separate CPU test dequantizes padded K=640 tiles
for Q8_0, IQ4_NL and Q2_0: it reproduces NaNs from poisoned scales times zero and
restores finite tails by clearing the guard. The 27 packet comparisons, policy
and FFN-640/checkpoint fixture, and ten Python tests pass in CPU Release. The
previous Q8_0 model test on two GPUs passed with NMSE 5.00462e-13, but it did not
exercise poisoned mixed-format reuse; that earlier pass is not validation of this
correction on the real model. The captured trace has no intermediate floating
point tensors, so the exact first corrupt GPU tensor remains unobserved.

The runner now rejects a candidate with at least 32 drafted tokens and zero
accepted tokens immediately after that response, retaining the response and
handoff. It no longer continues to call that run a completed performance result.
Rebuild both candidate directories, rerun `test-moe-stream` on ROCm0/ROCm1 and
the integrated model test, then use the same guarded A/B suite. The padding fix
has not been executed on a GPU in the implementation environment, and production
DFlash acceptance, output correctness and the performance/VRAM measurements must
be rechecked before merge. No change has been merged to `codex-compat`.

## 2026-10-30: keep streaming off the slow link and preserve generation demand

The `moe-stream-ab-20261030-095410-184302` archive identifies the candidate as
`58c330f`. Its cold uninstrumented response matches the baseline's completion hash
and 78/144 accepted drafts. Warm responses differ (83/130 versus 78/144), so decode
throughput is not an equal-work comparison. The candidate's warm prefill is still
3.00% slower, 436.78 versus 450.27 tokens/s; warm request wall time is 4.30% longer.
Removing D2D copies is insufficient to claim a speedup.

Host expert accounting in `02-measure` separates two problems:

| Expert traffic, GiB | Baseline | `58c330f` |
| --- | ---: | ---: |
| Cold large batches, ROCm0 | 146.7562 | 41.3732 |
| Cold large batches, ROCm1 | 0 | 30.5674 |
| Warm large batches, ROCm0 | 107.7442 | 37.6787 |
| Warm large batches, ROCm1 | 0 | 27.5710 |
| Warm small batches, both GPUs | 0.3389 | 13.3917 |
| Warm expert D2D | 43.3012 | 0 |

The baseline executes all large host-expert matmuls on ROCm0. Streaming instead
anchored 85 of 235 large layer batches on ROCm1. On this machine that GPU is behind
an x4 link. Whole-process profiler H2D duration sums change from 10.9025 to 4.3733
seconds on GPU0, but from 1.6968 to 11.4087 seconds on GPU1. These sums include
initialization, both requests and draft work; they are not isolated prefill costs
and must not be added to claim an end-to-end delta. The profiler timestamps also
have a different epoch from the runner's monotonic request timestamps; directly
joining them by multiplying request microseconds by 1000 would be incorrect.

For roughly half-resident groups and a fourfold link difference, the static
transfer estimate now selects ROCm0 streaming for ROCm1-owned large batches.
Local ROCm0 cache hits still compute in place. ROCm1's resident banks and budgets
are unchanged; decode/verify still uses them directly. Redirected prefill observes
demand but does not evict from or admit into that remote bank. It deliberately
streams more host bytes over the faster link instead of requiring slow-link
uploads to save bytes. This is not remote cache-hit execution or a peer-copy path.
The same bounded operation handles both placements. No attention arithmetic,
expert routing or dequantization kernel is changed.

The original all-token window also aged cold decode history out during the next
5121-token prefill. A separate small-batch progress window and priority preserve
that demand, while new local small requests can still replace stale residents.
This is a replacement-policy correction, not restoration of SLRU or a new cache.
Neither window repeatedly rewards speculative revisits to the same progress
stamp; neither claims to know which speculative tokens will eventually be accepted.

An offline replay consumes all 5220 routing/plan pairs from the candidate's
instrumented process, including initialization and both requests. Group capacities
are 1600/1066/3202/532/1333/265 on ROCm0 and 260/2089/782/1044/260/260 on ROCm1.
The unchanged policy reproduces every recorded hit/admission/temporary count
exactly. Replaying the correction with the same route histograms predicts warm
small-batch H2D of 3.3809 GiB instead of 13.3917 GiB, and large-batch H2D of
99.7135 GiB, entirely on ROCm0, instead of 65.2497 GiB split across both GPUs.
This uses the actual C++ planner, not a Python policy reimplementation. Small-batch
progress is reconstructed from changes of the recorded all-token clock; accepted
positions are not in the trace. Frozen routing, timing, overlap and future GPU
numerics are not simulated. These are policy/placement estimates, not measured
throughput or an acceptance guarantee. The warm decode estimate still exceeds the
baseline's 0.3389 GiB; that remaining gap is not hidden as a solved problem.

There are still no full expert work tensors or expert D2D copies on supported
streamed operations. The remote placement uses only tensor metadata views of the
already allocated scratch. If all ROCm1 groups redirect as estimated here, its
29.01 MiB scratch allocation disappears and ROCm0 keeps one shared 29.01 MiB bank.
Compute-buffer high-water marks and whole-process peak VRAM require a new hardware
run; the previous roughly 500.72 MiB main-GPU compute-buffer saving is not a new
measurement for this revision. Conservative attention-mask reservations remain.

Validation in the implementation environment: CPU Release with measurement enabled;
all three targeted CTest entries pass. The 27 poisoned-packet comparisons now also
exercise typed scratch views with empty resident packets and no separate working
bank. Policy tests cover topology crossover, full/unknown-link fallback, independent
progress clocks, retry suppression, decode protection through long prefills,
expiry and observation without remote-bank mutation. The model fixture retains
prefill/decode/verify and checkpoint/reset checks; GPU configurations now report
`redirected` counts separately. All eleven Python tests pass. Python type checking,
EditorConfig checks and `git diff --check` pass for the changed files.

No HIP SDK or GPU is present here. Rebuild both candidate directories, run the
existing two GPU packet fixtures and integrated model fixture, then use
`python3 scripts/moe-measure/stream-ab.py --profile`. The runner now verifies
scratch on every announced **execution** device, including mixed local/redirected
groups, rather than assuming cache ownership equals execution placement. Missing
declarations retain the previous strict per-owner check; working-tensor fallback
and zero-acceptance gates remain. Check that `ROCm1 prefill execution = ROCm0` is
announced on this machine and inspect the model fixture's `redirected` count.
Measure uninstrumented speed first, then transfer time, staging/compute memory and
DFlash acceptance. This revision has not been merged to `codex-compat`.

## PCIe path correction after the 120144 handoff

The `20261030-120144-995261` hardware run used `4a73011e7` but announced local
prefill on both GPUs. Its results do not validate redirected execution. The
separate latency-demand window did work: warm small-batch expert H2D was 0.2901
GiB, versus 13.3917 GiB in the previous candidate run. Total warm expert H2D was
63.0767 GiB and expert D2D was zero. The uninstrumented warm prefill changed from
439.21 to 457.53 tokens/s; completion hashes and speculative work differ, so this
is not an equal-work decode comparison or a redirection speedup.

The old link reader examined only the GPU endpoint. That misses an upstream
chipset/switch restriction when the endpoint itself has a wider link. The reader
now follows the resolved sysfs hierarchy and uses its narrowest link estimate.
A synthetic sysfs fixture reproduces identical x16 GPU endpoints with an x4
upstream connection on one path, verifies the resulting redirect decision, and
covers a slower intermediate bridge, idle speed, non-C numeric locales, malformed
PCI IDs and missing/invalid parent attributes. No card names, PCI addresses or
forced placement rules are added to runtime dispatch.

The handoff does not contain the host's sysfs attributes. It therefore cannot
establish whether this endpoint-only blind spot, unreadable attributes or another
topology estimate caused the inactive placement on that host. The startup link
lines now make that distinction inspectable. The A/B runner requires known link
data and placement declarations for multi-GPU cache runs and prints the actual
owner-to-execution map. Unknown topology still falls back locally in the runtime;
only the controlled multi-GPU benchmark rejects it instead of silently accepting
an unverified placement experiment.

Continue with the existing 5121-token / 128-token A/B fixture. For the reported
x16/x4 configuration, verify `ROCm1 prefill execution = ROCm0` and the corresponding
`Candidate-Prefill` map. Small decode/verify placement, KV/attention ownership,
resident budgets, packet arithmetic and cache admission are unchanged. No longer
context benchmark or new runtime control is introduced in this correction.

Validation: CPU Release build with measurement enabled; all three targeted CTest
entries pass (policy/topology, model prefill/checkpoint/reset, streamed packets).
The CPU model logits remain exact (NMSE 0); the 27 packet cases retain a worst
NMSE of 7.18e-15 and maximum absolute error of 3.06e-5. All twelve measurement-runner
Python tests, targeted Python type checks, EditorConfig, model-naming checks and
`git diff --check` pass.

HIP compilation, the redirected GPU model fixture, output/acceptance consistency,
activation-transfer overhead, new VRAM peaks and actual throughput remain to be
checked on gfx1100/gfx1101; the implementation environment has no HIP SDK or GPU.

## Combined decode/verify and upload pipeline package

This package starts from `21ddbbac` and keeps its placement and demand policy.
The supplied `20261030-130143-635571` archive finally announces ROCm0 prefill for
both owners. Its uninstrumented warm prefill is 549.19 versus 444.65 tokens/s;
the decode comparison has different completion hashes and speculative work.
In `02-measure/warm`, all 2490 routing histograms, completion hashes and DFlash
counts match. There, prefill is 549.70 versus 466.56 tokens/s and generation is
38.10 versus 37.28 tokens/s. Measurement-build timings are not uninstrumented
throughput results.

That measured warm candidate has 2197 all-hit small requests out of 2255, with
89.19 ms of CPU routing wait and 7.13 ms of host planning. Deleting those times
alone cannot explain a large multiple of its 3.33-second generation time. Larger
benefits would have to come from less graph fragmentation or better verify work
reuse and must be measured. The profiler's simultaneous device-allocation peak
falls by 471.71 MiB on that older revision, while resident cache sizes stay fixed.
The cold baseline is substantially slower than its next same-binary run, so the
large cold headline delta is not credited as an implementation speedup.

### Device slot lookup and asynchronous admission

For supported streaming groups with one to eight queries and at most 256 selected
IDs, `MOE_CACHE_LOOKUP` replaces the host map-copy boundary. The layer's I32 slot
map resides on its owning GPU. A wave32 block gathers physical slots and publishes
the logical IDs and a miss bit into a small coherent mapped notification.

An all-hit lookup continues without a CPU routing readback or admission decision.
A worker later observes demand using the same window policy. A delayed hit can
update demand but cannot revive a slot already evicted by a later request. A miss
waits on an HSA signal while the worker pins existing hits, admits the missing
experts directly into their final banks and uploads the group's complete mapping,
including invalidations of evicted experts in other layers. The upload stream
releases the signal after those copies; the GPU then re-gathers the slots.
There is no GPU spin kernel and no CPU stream synchronization inside admission.
The next layer cannot publish a miss before the preceding layer's readers finish.

The wait is an explicit HIP BatchMemOp graph node during capture. A plain
`hipStreamWaitValue64` call is not automatically captured. Its parameter arrays,
signals and notifications outlive all graphs; destruction clears those graphs
before releasing them. The implementation requires HIP 6.4 or newer, checks the
device stream-wait capability and probes an already-signaled graph wait at
initialization. Unavailable support retains host routing. Context synchronization,
batch/clock changes, graph recovery and destruction drain observers before reuse.
Each enabled GPU has one worker, asleep outside small-batch execution. This is
cache metadata, not another weight cache or a speculative expert predictor.

### Bounded transfer/compute overlap

Each prefill execution GPU now alternates two 32-expert staging lanes instead of
one. A separate nonblocking upload stream waits for a lane's consumed event,
uploads weights and descriptors, and records its ready event. MMQ/MMB wait for
ready, clear the same quantized read guard, compute, and record consumed. Empty
packets are handled too. The next lane can upload while the current packet runs;
reuse across layers and gate/up/down banks remains bounded and event ordered.

Uploads use the existing backend's malloc-stage pool and mutex, including its
mmap workaround and optional tensor filter. There is no second host-stage pool.
On the supplied placement, another lane adds approximately 29.01 MiB on ROCm0,
plus less than 0.1 MiB of device maps across both GPUs; the resident banks remain
unchanged. This is an allocation-size estimate, not a new measured peak saving.
ROCm1 needs no large-packet staging when all its prefill groups redirect to ROCm0.
Large-batch routing still reads back IDs once per layer. Transfers of the next
layer's experts cannot precede knowledge of its router result.

### Verify dispatch after hardware profiling

The grouped RDNA3 verify candidate from d9c17ce was numerically correct after the fractional-scale correction, but slower on both physical GPUs. It has been removed together with its dispatch override. Five to eight query operations again use the existing per-type MMVQ/MMQ/MMB selection. In particular, IQ4_NL/IQ4_XS retain native MMVQ through six queries, while IQ2_S/IQ3_S/IQ3_XXS retain their four-query limit. The GPU slot lookup, bounded upload pipeline and corrected shared dot products remain in use. No inactive candidate or runtime switch is retained.

The normal draft width stays unchanged. `decode-ab.py` runs the existing A/B
fixture at draft maxima 3, 5 and 7, corresponding to target verify widths up to
4, 6 and 8. Each width is applied equally to baseline and candidate; all other
Fish/JSON settings stay unchanged. No application flag or environment switch is
introduced. The existing main-model mixed quantization remains native.

### Validation and the next hardware round

The GPU packet fixture also uses both lanes, poisons every reused lane, and checks
the final partial guard. Twenty added real-shape verify cases compare the ordinary
verify dispatch against streamed MMQ/MMB on exactly Q8/BF16-representable activations.
A separate two-layer fixture runs the same lookup graph twelve times without an
evaluation callback, exercising cold admission, all-hit continuation, eviction,
duplicate/unsorted IDs and graph capture/replay against a dequantized CPU oracle.
Unsupported signal graphs print a skip; they do not pass as device-lookup proof.
The integrated model fixture adds widths five and seven to its existing prefill,
checkpoint and sequence-reset comparisons. CPU policy tests cover a delayed
hit observation after eviction without changing residency.

New HIP kernels, graph waits, event ordering, production logits/DFlash acceptance,
Vision/save/restore/rollback compatibility, end-to-end gains and peak VRAM are
still GPU tests to be run on the two physical cards. Existing attention, KV,
prompt-tail ingest, SSD cache and Vulkan construction are unchanged. CPU model
equality does not validate the new device kernels.

Keep the baseline binaries. Rebuild both candidate build directories and the two
test targets with the normal ROCm configuration, then run:

```sh
./build-rocm-moe-stream-reference/bin/test-moe-stream --backend ROCm0
./build-rocm-moe-stream-reference/bin/test-moe-stream --backend ROCm1
./build-rocm-moe-stream-reference/bin/test-llama-archs --test-moe-cache-prefill -s 42
python3 scripts/moe-measure/decode-ab.py --profile
```

The suite forwards the existing A/B runner options, writes one handoff for all
widths, and preserves partial results on failure. It requires device routing and
the transfer pipeline on all cache GPUs, retains placement/working-copy and
zero-acceptance gates, and stops at the first failing width. Runtime fallback
remains safe; this controlled test rejects it instead of calling it a successful
three-part benchmark. `--draft-widths 3` can run the existing width alone.
Host reports now separate device all-hit layers, worker planning time and legacy
CPU routing wait. A zero CPU routing wait does not mean a zero-duration GPU signal
wait or zero observer cost. Inspect profiler kernel/transfer overlap and actual
allocation lifetimes separately from host enqueue measurements.

Implementation checks: CPU Release with measurement enabled passes the three
targeted CTest entries. The CPU model logits remain exact (NMSE 0); the 27 packet
cases have worst NMSE 7.18e-15 and maximum absolute difference 3.06e-5. Release
without measurement also builds `llama` and the RPC backend; RPC's operation-count
protocol patch is bumped to 8.0.3, and both process-local operations stay rejected.
All 18 measurement/runner tests, targeted Python type checks, EditorConfig and
`git diff --check` pass. These results do not execute device lookup or the new
verify kernel; the dedicated GPU fixtures are provided for the local hardware run.

The final Release HIP build succeeds for both `gfx1100;gfx1101`, including
`ggml-hip`, `llama-server`, `test-moe-stream` and `test-llama-archs`. Compilation
uses the official ROCm 10.3.0a20261030 SDK (Clang 24), not the production machine's
ROCm 10.1. That SDK warns about the context-query API's deprecation; it remains
available for the HIP 6.4-compatible BatchMemOp setup. No physical GPU is exposed
in this environment, so the new GPU fixtures are compiled but not executed.

## IQ2_S / IQ3_XXS fractional-scale correction after d9c17ce

The first physical gfx1100 run passes all 27 previous packet cases with NMSE 0.
The new five-query IQ2_S case then aborts at NMSE 2.75919393e-10, above the
unchanged 1e-10 limit. IQ3_XXS also has a larger discrepancy than the other formats
(about 1.04e-11). ROCm1 was interrupted before completion, and the later device
lookup and integrated model checks are not validated by that partial run.

The shared native MMVQ helpers, also used by ordinary decode, truncate fractional
scales in integer arithmetic. IQ2_S computes
`(s0*l0 + s1*l1 + (s0+s1)/2)/4` as integers. Its dequantization instead requires
`(s0*(2*l0+1) + s1*(2*l1+1))/8`. IQ3_XXS similarly needs `(2*l+1)*s/4`
rather than integer-truncated `(l*s + s/2)/2`. The MMQ path retains these fractions.
The correction keeps the scaled numerator integral and applies 1/8 or 1/4 in
FP32 after conversion. Dot-product sums and scale factors fit the existing signed
24-bit multiply operands and 32-bit accumulator. No test tolerance is widened.

A CPU reproduction uses the fixture's seed 47, K=2560, M=640, five queries and the
same selected experts and exactly representable Q8 activations. It emulates each
old/corrected native dot and wave32 reduction and compares against CPU-dequantized
weights with double-precision accumulation:

| Format | Old helper NMSE | Corrected helper NMSE |
| --- | ---: | ---: |
| IQ2_S | 2.75835298e-10 | 5.46604166e-15 |
| IQ3_XXS | 1.03850740e-11 | 5.44969173e-15 |

These are CPU emulation results, not new GPU measurements. The small difference
from the reported GPU pairwise error also includes MMQ's FP32 accumulation order.
The correction affects both existing decode and grouped verify for these two
formats, so their numerical output can change slightly from the old baseline.
Router selection, cache policy, attention arithmetic and runtime settings are
unchanged. Production logits and DFlash acceptance still need hardware validation.

The twenty small-verify GPU cases now also run each query separately through the
ordinary one-token operation. They compare grouped verify, streamed MMQ and
single-token decode independently with a CPU dequantization/double-accumulation
oracle, and compare decode with verify. All NMSE limits remain 1e-10. Diagnostic
values are flushed before asserting so redirected failure logs retain them.

Local validation: the Release CPU `test-moe-stream` CTest passes all 27 packet cases and the padding guard check (22.55 s). The HIP Release build compiles and links `llama-server`, `test-moe-stream` and `test-llama-archs` for gfx1100 and gfx1101 with ROCm 10.3.0a20261030 (Clang 24). No physical GPU is available here; the corrected GPU arithmetic, device lookup, integrated model behavior and performance remain to be checked with the production ROCm 10.1 setup.

Rebuild both candidate directories after pulling this correction. In Fish, this
block captures stdout and stderr from all three tests and stops after a failure:

```fish
begin
    ./build-rocm-moe-stream-reference/bin/test-moe-stream --backend ROCm0
    and ./build-rocm-moe-stream-reference/bin/test-moe-stream --backend ROCm1
    and ./build-rocm-moe-stream-reference/bin/test-llama-archs --test-moe-cache-prefill -s 42
end > test.log 2>&1
```

Run the performance suite only after the correctness tests complete successfully.

## Physical GPU results after f2efb69 and verify rollback

The supplied `moe-decode-ab-20261030-202448-724033-handoff.tar.gz` completes all draft widths. `test(2).log` validates 47 packet cases per GPU, including twenty verify cases with independent CPU and one-token references. Worst pairwise NMSE is 5.86442089e-14; the worst CPU-oracle NMSE is 5.35913598e-14. The device-lookup fixture passes on both gfx1100 and gfx1101 with NMSE 2.5884e-15, including twelve graph executions and cross-layer eviction. The integrated model fixture passes checkpoint/reset and cache-budget fallback comparisons with worst logit NMSE 5.00462e-13. Its summaries report zero redirected operations, so it is not proof of the remote-owner prefill path in the production model.

The uninstrumented warm measurements show that faster prefill is being offset by a verify regression:

| Draft maximum | Baseline prefill t/s | Candidate prefill t/s | Baseline decode t/s | Candidate decode t/s |
| --- | ---: | ---: | ---: | ---: |
| 3 | 455.70 | 544.07 | 33.72 | 32.41 |
| 5 | 474.22 | 580.39 | 34.83 | 21.69 |
| 7 | 479.51 | 594.46 | 28.70 | 18.07 |

The draft-7 profiler records an average IQ4_NL MMQ kernel duration of 173.69 us on gfx1100 and 187.61 us on gfx1101 in the baseline. The new grouped verify kernel averages 929.53 us and 1406.87 us respectively. These are per-call averages from the mixed cold/warm runs, not a fixed-routing microbenchmark. The new verify kernels sum to 4.79 GPU-seconds at draft 5 and 6.04 GPU-seconds at draft 7 across both GPUs. Summed GPU durations can overlap and must not be subtracted directly from request time.

The code repeats leader/deduplication scans for each four-row block and serializes matching query dots while staging weights in LDS. This is a plausible source of the measured cost; the traces establish the regression, not a breakdown of LDS stalls versus instruction cost. Removing the candidate restores the previous dispatch exactly, including the native kernels it had also overridden for already-supported types.

All completion hashes differ between baseline and candidate in this suite. Draft-5 warm acceptance falls from 86/197 to 76/251, while draft-7 warm acceptance rises from 85/287 to 89/260 despite slower decode. The kernel regression therefore cannot be attributed to lower acceptance alone. The request uses temperature 0.7 and seed 42; the changed generation trajectories do not establish production-logit equivalence. The synthetic numerical tests passing does not close that production validation requirement.

GPU-side cache resolution is active and its device waits are still measurable. A zero host routing wait in the report does not mean zero GPU waiting. Those waits and cache demand should be assessed again after removing the verify regression, before changing the admission policy or synchronization design. The existing prefill placement and both staging lanes are preserved.

For the next hardware comparison, retain the baseline binaries and rebuild only the candidate directories. Run the same numerical tests first, then `python3 scripts/moe-measure/decode-ab.py --draft-widths 5 --profile`. This focuses the next comparison on the clearest regression. No speedup after the rollback has been measured yet.

Local rollback validation: the CPU Release targets `test-moe-stream` and `test-llama-archs` build successfully. The three targeted CTest entries `test-moe-stream`, `test-moe-cache-policy` and `test-moe-cache-prefill` pass (22.47 s total); the GPU cases skip without hardware. Source comparison confirms that MMVQ and the `MUL_MAT_ID` dispatcher exactly match their pre-candidate implementations, while the f2efb69 numerical correction and regression tests remain unchanged. A HIP compiler is unavailable in the current workspace, so neither a fresh HIP build nor physical GPU execution of this rollback is claimed.

## Completion differences: diagnostic before further decode changes

The draft-3 handoff `moe-decode-ab-20261030-233708-744468` contains three distinct completion hashes within the baseline's six requests, and three within the candidate's six requests. Recomputing the hashes from the archived responses confirms the implementation: they cover message/text/finish reason, including reasoning, and exclude response IDs and timings. The differences are real generated text. Both builds vary between requests; the baseline already does so without the new cache execution.

The request uses temperature 0.7 and seed 42, while the Fish template also enables dynamic temperature. A fixed seed does not establish identical rendered prompt tokens or bitwise-identical target probabilities. The candidate additionally contains the intentional IQ2_S/IQ3_XXS fractional-scale correction that the old baseline lacks. These are relevant differences, not a proven explanation of the first changed token. Existing synthetic model coverage uses Q8_0 experts and does not establish mixed-IQ production-model equivalence.

Keep draft maximum 3 while investigating this. The focused diagnostic uses the existing uninstrumented baseline and candidate binaries, the same positive MoE cache budgets and the reference Fish template:

```sh
git pull --ff-only
python3 scripts/moe-measure/completion_diagnose.py
```

No CMake rebuild or packet test is needed for this Python-only addition. Stop an existing server on port 8080 before running it. The original Fish template and request are not edited. Generated diagnostic launchers set the existing draft maximum to 3; sampling changes are confined to the diagnostic HTTP requests. There are no new application flags or environment variables.

The script renders and tokenizes the original chat request on each build, checks that the prompt is identical, then sends explicit token IDs to the native completion endpoint. Each build generates twice with plain greedy sampling: temperature 0, dynamic range 0 and only the temperature sampler. Raw output token IDs and the pre-sampling top-32 log probabilities are saved. This removes the chat response parser from the token comparison and checks repeatability within each build as well as differences across builds. Library hashes, executable hashes and server properties are recorded too.

It then starts fresh servers and replays the same baseline token prefixes on both builds. Probes cover the prompt boundary and the earliest observed token divergence; if greedy outputs match, the second probe is at position 32 or the last available token. Each probe predicts one token and verifies that no speculative drafts were generated, separating a fixed-prefix target check from free-running verify behavior. Probabilities from generated sequences are compared only through the first differing token, while the preceding token prefix is still identical.

The result is `report.json` plus one handoff archive. A complete diagnostic means data collection succeeded, not that model equivalence passed. Top-32 log-probability differences and top-1 margins are evidence for investigation; they are neither full-vocabulary logit NMSE nor a quality verdict. These runs force CPU probability collection and different sampling controls, so their durations are not performance measurements. Real-model GPU execution remains a hardware task; there are no further decode optimizations in this diagnostic change.

Local checks: nine focused Python tests pass, covering greedy controls, unchanged cache budgets, generated-token/probability alignment, prompt mismatch rejection, fixed baseline prefix replay, comparison stopping at the first different token, and partial handoff preservation. Targeted `ty`, EditorConfig and `git diff --check` pass. The native HTTP/model execution has not been run here because the production models and GPUs are unavailable.

## Greedy diagnostic results and single-token control

The supplied `moe-completion-diagnose-20261031-001111-941090` handoff verifies the same rendered prompt and 5121 prompt token IDs on builds 82478ef and 579880d. Every returned token is an argmax of its reported raw probabilities with temperature 0, dynamic temperature 0, mirostat 0 and only the temperature sampler. The baseline's two generations first differ at zero-based position 125; the candidate's first differ at position 7. Candidate repeat 1 has exactly the same 128 output token IDs as baseline repeat 0.

The early candidate divergence follows the shared text `The user has provided 40`. Repeat 0 ranks ` identical` at 44.67% and ` cases` at 42.88%; repeat 1 ranks ` cases` at 44.34% and ` identical` at 41.61%. The old baseline also reverses two close leading choices at its position 125. These are actual token changes with the same preceding tokens, not response-format or random-sampling differences. Close margins explain how a probability change becomes a different completion; they do not explain the probability change itself.

Raw probability rows already differ at the first token after prefill, including within each build. Both fresh, fixed-prefix replay probes generate the same next token across builds and report no draft tokens, but their top-32 log probabilities differ. At prefix 7 the baseline's replay chooses ` identical`, whereas both of its free-running generations choose ` cases` at that same prefix. Replay recomputes the prefix in prefill and changes the final prompt batch from 1 to 8 tokens; free-running generation uses incremental decode/verify state. This localizes a comparison to execution path or state differences, without identifying the responsible operator or proving a rollback defect. The summary's maximum shared log-probability delta is not a full-vocabulary logit error and can come from low-probability candidates.

The next control reuses the same binaries, prompt and cache budgets but sets the existing server draft maximum to 0:

```sh
git pull --ff-only
python3 scripts/moe-measure/completion_diagnose.py --draft-max 0
```

No build, packet test or profiler is needed. The original launcher remains unchanged. The draft model and its target-feature capture remain configured; this is a control without draft tokens and speculative target verify batches, not a test that removes all DFlash infrastructure. Draft projection/ingestion can still run, so its timings are not a non-speculative performance benchmark. The diagnostic rejects any response reporting nonzero draft tokens, including the repeated generations, and archives incomplete results on failure.

Within-build repeatability in this control distinguishes variation that persists without speculative verify from variation that needs the speculative execution path. It does not alone prove the cache correct. Cross-build differences still include the IQ2_S/IQ3_XXS fractional-scale correction absent from the old baseline, changed prefill placement and different matrix-multiply paths. No kernel, attention arithmetic or cache-policy change is made based on the greedy handoff. Physical execution of the zero-draft control remains open; eleven focused Python tests cover collection, validation and failure handling locally.

## Zero-draft results and real-score TopK isolation

The supplied `moe-completion-diagnose-20261031-002803-346227` handoff completes draft maximum 0 with identical prompt and binary/library provenance to the preceding draft-3 diagnostic. Both repeats within each build produce the same 128 token IDs. Their raw probabilities still differ, including at the first post-prefill token. Removing speculative verify therefore did not eliminate numerical variation; two equal completions are not proof of reproducible logits.

The first cross-build token difference moves to position 125 (zero-based), after 125 identical generated tokens. Baseline repeats choose ` single` at 38.76% and 37.40%; candidate repeats choose ` substantive` at 38.12% and 39.06%. Fixed-prefix prefill replay at 125 chooses ` single` in both builds (37.26% and 38.04%). The candidate still has a prefill/incremental-path difference without speculative verify. This does not identify its cause, and the old baseline's missing fractional-scale correction remains a cross-build confounder.

One concrete shared candidate is ordinary HIP `TOP_K`, before `QSA_SELECT`: `common.cuh` excludes HIP from `GGML_CUDA_USE_CUB`, and `top-k.cu` uses radix selection above the default 1024-column bitonic threshold. At the cutoff, `atomicAdd(equal_count)` admits the first arriving tied cells. The kernel permits different index sets with the same selected score values, which existing backend tests intentionally accept. For QSA, different finite-score pools can contain different K/V values, so valid but changing tied selections can affect attention. Pool padding adds an unused block; the current 5121-token, four-cell-pool prompt reaches this radix path. Neither the presence of finite cutoff ties in the production scores nor their responsibility for the observed logits has yet been measured. Reordering the same pool set or changing choices only among `-inf` pools cannot explain a QSA mask change.

The new `test-qsa-topk-trace` executable uses the existing model/cache/backend arguments and exact prompt token IDs from the zero-draft diagnostic. It runs two prefills in one process with fresh model memory and retained expert-cache state. It collects only named QSA TopK nodes through the existing evaluation callback, validates index bounds, uniqueness and ranking, and records per-row SHA-256 fingerprints of scores and canonical visible pool sets. The second pass distinguishes changed score inputs from changed visible sets with matching score fingerprints. It makes no speculative drafts or decode calls. The configured draft model and target-feature capture are initialized; draft ingestion/projection and server scheduling are not reproduced.

For the first batch with a finite ambiguous cutoff on each observed device, it also copies the entire score matrix unchanged to an independent tensor on that same device and repeats ordinary TopK eight times. It compares visible sets against the original operation and between replays, then byte-compares the frozen input against its original copy. Each input is bounded to 32 MiB; prompt length is bounded to 8192 and rows to 1024. Only small fingerprints and summary counters are retained across calls, and tensor values are not dumped to the log.

Run this focused diagnostic on the existing configured HIP build; a separate server must be stopped first:

```fish
git pull --ff-only
and begin
    set -lx ROCM_PATH $HOME/opt/rocm-10.1
    set -lx HIP_PATH $ROCM_PATH
    set -lx LD_LIBRARY_PATH $ROCM_PATH/lib $LD_LIBRARY_PATH
    cmake -S . -B build-rocm-moe-stream-reference
    and cmake --build build-rocm-moe-stream-reference --target test-qsa-topk-trace -j28
    and python3 scripts/moe-measure/qsa_topk_diagnose.py --source-run results/moe-completion-diagnose-20261031-002803-346227
end
```

The Python wrapper preserves the original Fish file and cache budgets. It prints a short summary and creates a handoff containing `events.json`, `report.json`, the generated launcher, prompt IDs and the native log. Nonzero exit, missing coverage, no ROCm coverage or incomplete frozen-input replay is a diagnostic failure. Partial logs are archived on failure. No additional benchmark or `test-moe-stream` run is requested.

Interpretation: `matching_score_hash_changed_visible_set_rows > 0` or changed sets on byte-verified frozen inputs demonstrate selection variability for the captured scores. Finite ambiguous cutoffs without observed changed sets show a possibility, not causation. No finite cutoff ambiguity would reject this candidate for the captured inputs. Changed score rows already have an upstream difference that TopK did not create. The evaluation callback introduces synchronization and can prevent cross-node fusion; absence of drift under this instrumentation does not exclude a race or other uninstrumented execution-path issue. These are numerical diagnostics, not performance measurements, and do not establish production logits or DFlash acceptance equivalence.

Local validation: CPU Release build and the focused CTest pass. Native self-tests cover finite and masked ties, signed zero, output-order independence, invalid selections, eight CPU frozen-input executions, and the distinction between changed inputs and changed selections. Six Python tests cover exact prompt/budget reuse, provenance checks, trace completeness, absence of ROCm coverage, frozen-input validation, timeout cleanup restricted to the owned process group and partial handoff preservation. Targeted type/style checks pass. Physical gfx1100/gfx1101 model execution and the TopK hypothesis remain open. No production backend, kernel, attention arithmetic or cache policy is changed.

### Model-resolution startup fix

The supplied `qsa-topk-diagnose-20261031-010355-855240` handoff stops before model loading: the native target path is empty. `common_params_parse(..., LLAMA_EXAMPLE_SERVER)` deliberately defers HF/model resolution to the caller, and the standalone diagnostic omitted that step. The tool now calls the same `common_models_handler_init/apply` pair used by the server, resolving both the target and explicit draft reference through the existing cache/download mechanism before initializing their contexts. Empty resolved paths are rejected explicitly. This changes diagnostic startup only; no QSA measurement was obtained from the failed run.

The wrapper also includes the native diagnostic error in its failure summary when available, preserving the full log and partial handoff. Rebuild only `test-qsa-topk-trace` and rerun the same Python command; CMake reconfiguration, server rebuild and a new completion diagnostic are unnecessary.

Validation: the CPU Release target builds; the focused `test-qsa-topk-trace` and existing `test-model-resolution` CTests pass. A native offline startup smoke test with an isolated synthetic HF cache reaches the intentionally invalid target GGUF file after resolving model references, rather than failing with an empty model path. It performs no inference or downloads. Six Python tests, targeted type/style checks and `git diff --check` pass. Real-model HIP execution remains open.

### Large-batch lookup service deadlock

The supplied `qsa-topk-diagnose-20261031-011039-171481` reaches all twelve QSA layers of the first 1024-token batch, then times out. Both GPUs report enabled device routing. The twelve captured operations have 320 columns and k=320; none has a finite ambiguous cutoff or starts a frozen replay. They do not test the later radix-selection hypothesis. The reported prolonged GPU load is consistent with a device wait, not evidence of slow TopK sorting.

Source inspection identifies a liveness bug at the next stage. The diagnostic requests one output per 1024-token batch. Qwen4exp crops its last-layer FFN to that output row, so this operation qualifies for device cache lookup even though its parent ubatch has 1024 tokens. `begin_compute()` previously started admission workers only when the parent batch had at most eight tokens. A cold last-layer lookup therefore publishes a miss and waits for a ready signal, while its servicing worker stays parked. This also affects other callers producing such mixed large/small graphs; it is not a TopK arithmetic defect. Confirmation that this explains the physical hang requires the next GPU run.

`begin_compute()` now receives the executing graph. The existing at-most-eight-token fast path is retained; larger graphs also activate the existing services when they contain `MOE_CACHE_LOOKUP`. Large graphs without that operation keep the workers parked. No graph topology, kernel, weight-transfer mechanism, cache admission rule or numerical computation is changed. The synthetic MoE model test now starts with a cold 64-token prefill requesting one output, checks its logits against the uncached model and records cropped device-lookup coverage. When last-layer device lookup is available, later decode lookup coverage also requires that the cropped prefill path was covered. CPU execution validates the model/cropping comparison but cannot validate the HIP service handshake.

The diagnostic's overall wait is reduced from 900 to 180 seconds. On timeout it now preserves complete TopK records in `events.json` and records the last captured event in `report.json`, skipping an unfinished final log line. Process-group cleanup remains restricted to its own child. This timeout bounds a failed diagnostic; it is not a kernel performance threshold.

Rebuild the existing `test-qsa-topk-trace` target and rerun the same `qsa_topk_diagnose.py` command. Its library dependency rebuild includes the cache-service fix. A full backend test suite or A/B performance run is not needed for this next physical check. The original completion drift and DFlash acceptance comparison remain unresolved; fixing this deadlock does not establish numerical equivalence.

Local validation: both CPU Release targets build. The focused `test-moe-cache-prefill` and `test-qsa-topk-trace` CTests pass (1.58 seconds total). The model fixture reports worst logit NMSE 0 with checkpoint/reset checks passing; its device/cropped lookup counters are both zero because HIP execution is skipped. Seven Python tests pass, including retention of partial events after timeout and a truncated final log record. Parsing the actual failed GPU log recovers twelve records on ROCm0/ROCm1 at position 0, ending at layer 47, with no frozen replay. Targeted type/style checks and `git diff --check` pass. A fresh HIP build, the GPU cropped-lookup regression and a completed physical-model trace remain open.

### Physical TopK result and deterministic cutoff selection

The supplied `qsa-topk-diagnose-20261031-093107-400368` completes in about 51 seconds on gfx1100/gfx1101. Both prefills reach all 72 QSA TopK calls with identical score fingerprints and visible pool sets. The prior large-batch lookup hang is no longer reproduced. The ten finite ambiguous rows reported across both passes are five distinct row occurrences per pass, at prompt position 3072 with 1088 columns and k=512: one each at layers 11 and 15 on ROCm0, and two at layer 31 plus one at layer 35 on ROCm1. Each cutoff has two equal scores competing for one remaining place.

Frozen-input replay does change selections on both devices. At layer 11 on ROCm0, one row changes between repeats 0 and 1. At layer 31 on ROCm1, up to two rows change between consecutive repeats. Every replay byte-checks its unchanged source matrix; ranking, index bounds and uniqueness pass. This establishes ordinary HIP radix TopK selection variability for actual finite QSA scores. It does not establish that this is the only cause of the earlier completion/probability differences. The two instrumented prefills did not drift, and this run contains no decode, verify or acceptance measurement.

The custom CUDA/HIP radix gather now resolves identical cutoff keys by ascending source-column index. Histogram workgroups cover contiguous column ranges. The last histogram already counts exact cutoff keys per range; these counts provide each workgroup's starting tie rank. Within a workgroup, the existing physical-wave prefix-sum helper and a small shared wave-count array assign each tie its deterministic rank. The first remaining k ranks are retained. Columns above the cutoff retain their existing unordered atomic gather, so output order remains unspecified, while the selected set is stable. QSA_SELECT sorts that set before attention, and TOP_K_MASK is independent of its ordering. This does not assume anything about KV-cell order or change unequal-score ranking, score arithmetic, pool expansion, attention or cache state. Signed-zero key ordering is retained.

No device allocation or kernel launch is added; the previous global tie counter is removed. Workgroup barriers for tie compaction are internal to the existing gather kernel, with no host synchronization. Existing dispatch thresholds, bitonic and NVIDIA CUB paths, CPU, Vulkan and application controls are unchanged. This is a correctness fix, not a demonstrated throughput improvement. Its HIP build, physical cutoff tests, real-score replay, uninstrumented repeatability and performance still require hardware validation; the local CPU build cannot compile or execute the changed GPU kernel.

The existing trace executable also provides a model-free ROCm cutoff regression. Four shapes cover the observed 1088-column prefill, unaligned tails, one-row selection and a 65537-column range spanning the 64-workgroup cap. Inputs include zero ties, masked ties, two finite cutoff ties, distinct positive/negative scores, many finite ties and fewer visible cells than k. Eight executions per shape must preserve ranking, uniqueness, input bytes and the exact independently sorted lower-index cutoff set. The default CPU self-test validates these fixtures, ranking and replay stability but explicitly skips the HIP-specific lower-index requirement. Two matching tie cases are added to test-backend-ops; its cross-backend check retains the ordinary unordered/tied-score contract.

On the existing configured HIP build, stop any running model server and run:

```fish
git pull --ff-only
and cmake --build build-rocm-moe-stream-reference --target llama-server test-qsa-topk-trace -j28
and ./build-rocm-moe-stream-reference/bin/test-qsa-topk-trace --backend ROCm0
and ./build-rocm-moe-stream-reference/bin/test-qsa-topk-trace --backend ROCm1
and python3 scripts/moe-measure/qsa_topk_diagnose.py --source-run results/moe-completion-diagnose-20261031-002803-346227
```

The finite-ambiguity count need not become zero; equal scores remain equal. Both frozen change counters should be zero. After that check, `python3 scripts/moe-measure/completion_diagnose.py --draft-max 0` tests uninstrumented target repeatability, before repeating with draft maximum 3. The old baseline still lacks the intentional IQ2_S/IQ3_XXS fractional-scale correction and this cutoff fix, so cross-build differences alone cannot establish a MoE-cache regression. No additional performance optimization is bundled with this correction.

Local validation: CPU Release builds of test-qsa-topk-trace and test-backend-ops pass. The focused trace CTest passes in 0.54 seconds, including all four synthetic shape fixtures and their eight CPU replays. The two added backend-op cases pass on CPU. EditorConfig and git diff --check pass. There is no HIP/CUDA SDK or physical GPU here: the modified device kernel is neither compiled nor executed locally. The existing Python wrapper is unchanged, and no model generation or performance suite is run locally.

## 2026-11-01: corrected reference for the next comparison

The supplied physical gfx1100/gfx1101 cutoff regressions pass all four shapes and eight repetitions per shape on each GPU. The real-score trace `qsa-topk-diagnose-20261031-095514-825318` completes both 72-call prefills with unchanged scores and visible sets; all sixteen frozen executions now retain the original selection. The ten finite ambiguous rows still exist, as expected. This validates the cutoff correction for the captured cases, not every model input or its performance.

The two subsequent completion handoffs (`095808-995396`, draft maximum 0, and `100407-664427`, maximum 3) use identical binaries and the same 5121 prompt token IDs. Within the candidate, both repetitions have exactly equal 128 output token IDs and all reported top-32 probability rows. Draft 0 and draft 3 also choose the same 128 tokens. Both draft-3 candidate requests accept 87/118 draft tokens (73.73%), with forty DFlash rounds and mean output length 3.17. The historical baseline still varies between repetitions.

Decode and speculative verify are not established as numerically equivalent: 127 candidate probability rows differ between draft settings. The largest selected-token log-probability difference is 0.12464; the largest shared top-32 difference, 1.42216, concerns a very low-probability candidate. These are reported probabilities, not full-vocabulary logit NMSE. Candidate prefix replays at 0 and 125 are identical across the two draft settings. A numerically aligned legacy execution reference is needed before attributing remaining differences to the MoE changes.

### Reference scope

Reference branch: [diagnostics/moe-reference-correctness](https://github.com/inf9144/llama.cpp/tree/diagnostics/moe-reference-correctness), pinned commit `55ee91f6a6e63c1a47de748e2d20cbc0d6db7c34`, directly on `82478efb935b4e6a99946132553435d6f55fe8cb`.

Only two production files change: the fractional IQ2_S/IQ3_XXS scale correction in `vecdotq.cuh` and the deterministic radix cutoff in `top-k.cu`. Both complete files are byte-identical to the tested candidate at `45d51cb`. The existing `test-qsa-topk-trace.cpp` is copied unchanged and registered in CMake so that the same model-free cutoff regression can run against the reference's own libraries and headers. No MoE streaming operation, device lookup, window policy, cache placement, allocator or model/attention change is copied into the reference. Its original measurement hooks remain available and default-off.

The corrected binaries use new `build-rocm-moe-corrected-reference` and `build-rocm-moe-corrected-measure` directories. Historical build directories and Fish templates are not rewritten. The completion diagnostic accepts the same test-only `--baseline-reference` binary override already available in stream-ab.py. Default paths remain compatible with earlier runs. The A/B report no longer labels an arbitrary selected baseline as source commit 82478ef; executable hashes and startup logs remain the evidence for each selected build.

### Evening build and numerical controls

Run this from the root checkout on `feature/moe-cache-prefill`, with no model server occupying port 8080. The initial-cache script copies the matching candidate CMake compiler, architecture, compile/link flags, backend and llama options through CMake's cache reader. It omits package discovery, build/source paths and version metadata; it rejects embedded candidate build-directory paths and incompatible HIP/instrumentation settings. SDK discovery still uses the same ROCm 10.1 environment below. Existing candidate build caches must be present. The worktree check refuses a different revision or tracked edits instead of resetting an existing checkout.

```fish
begin
    set -lx ROCM_PATH $HOME/opt/rocm-10.1
    set -lx HIP_PATH $ROCM_PATH
    set -lx LD_LIBRARY_PATH $ROCM_PATH/lib $LD_LIBRARY_PATH

    git pull --ff-only
    and git fetch --no-tags origin diagnostics/moe-reference-correctness
    and begin
        if test -e ../llama-moe-reference-correctness
            test (git -C ../llama-moe-reference-correctness rev-parse HEAD) = 55ee91f6a6e63c1a47de748e2d20cbc0d6db7c34
            and git -C ../llama-moe-reference-correctness diff --quiet HEAD
        else
            git worktree add --detach ../llama-moe-reference-correctness 55ee91f6a6e63c1a47de748e2d20cbc0d6db7c34
        end
    end
    and python3 -m unittest discover -s scripts/moe-measure -p test_completion_diagnose.py
    and cmake -S ../llama-moe-reference-correctness -B build-rocm-moe-corrected-reference -G Ninja \
        -C scripts/moe-measure/reference-options.cmake -DLLAMA_BUILD_TESTS=ON
    and cmake --build build-rocm-moe-corrected-reference --target llama-server test-qsa-topk-trace ggml-hip -j28
    and ./build-rocm-moe-corrected-reference/bin/test-qsa-topk-trace --backend ROCm0
    and ./build-rocm-moe-corrected-reference/bin/test-qsa-topk-trace --backend ROCm1
    and python3 scripts/moe-measure/completion_diagnose.py --draft-max 0 \
        --baseline-reference build-rocm-moe-corrected-reference/bin/llama-server
    and python3 scripts/moe-measure/completion_diagnose.py --draft-max 3 \
        --baseline-reference build-rocm-moe-corrected-reference/bin/llama-server
end
```

This builds the corrected reference only and reuses the existing candidate executables. The source files affecting candidate execution are unchanged by this preparation. The Python tests use mocked HTTP/server calls and no GPU/model downloads. The two native checks load no model; each verifies four shapes with eight GPU repetitions. The two completion diagnostics then write separate handoff archives. A successful collection exit is not an equivalence verdict: return both archives to compare within-build stability, identical-prefix baseline/candidate probabilities and draft-0/draft-3 consistency before a performance run.

### Prepared profiler command after numerical review

The existing A/B suite already accepts separate baseline reference/measurement paths. After reviewing the numerical controls, build the corrected measurement binary with options from its matching candidate configuration, then profile only draft maximum 3:

```fish
begin
    set -lx ROCM_PATH $HOME/opt/rocm-10.1
    set -lx HIP_PATH $ROCM_PATH
    set -lx LD_LIBRARY_PATH $ROCM_PATH/lib $LD_LIBRARY_PATH
    cmake -S . -B build-rocm-moe-stream-measure -DLLAMA_MOE_MEASURE=ON
    and cmake --build build-rocm-moe-stream-measure --target llama-server ggml-hip -j28
    and cmake -S ../llama-moe-reference-correctness -B build-rocm-moe-corrected-measure -G Ninja \
        -C scripts/moe-measure/reference-options.cmake -DLLAMA_BUILD_TESTS=ON
    and cmake --build build-rocm-moe-corrected-measure --target llama-server ggml-hip -j28
    and python3 scripts/moe-measure/decode-ab.py --draft-widths 3 --profile \
        --baseline-reference build-rocm-moe-corrected-reference/bin/llama-server \
        --baseline-measure build-rocm-moe-corrected-measure/bin/llama-server
end
```

This performance step is separate from the initial controls. Keep existing model, cache budgets, prompt, batch and device settings unchanged. Throughput must be interpreted together with generated tokens and DFlash work; reference/candidate differences alone do not identify a waiting, transfer or kernel bottleneck.

Validation limits for this preparation: the execution environment is unavailable. No new Python/CMake execution, CPU/HIP build, type/style checker or physical GPU test has run here. Source inspection and repository comparisons verify the four-file reference scope, the two exact production-file matches and the unchanged copied test. Two unit cases are added for relative and absolute/space-containing baseline binary overrides. The native reference build, copied tests, CMake option transfer and new Python cases remain to be executed. The production branch is not merged.

## 2026-11-02: aligned correctness and first comparable throughput

Build instructions for this Ryzen 9 7950X setup use `-j28`. The earlier profiler recipe omitted rebuilding the candidate measurement binary; it is corrected above. The executable bit on `stream-ab.py`, lost during the previous tooling-only update, is restored.

### Numerical controls on gfx1100 and gfx1101

The corrected reference at `55ee91f` passes all four synthetic cutoff fixtures and eight repetitions per fixture on each GPU. Completion handoffs `moe-completion-diagnose-20261102-091419-294786` and `moe-completion-diagnose-20261102-091652-931861` compare that reference against the unchanged candidate executable at `45d51cb`. Both use the same 5121 prompt token IDs. At draft maximum 0 and 3, both repetitions and both builds produce exactly the same 128 token IDs and all recorded top-32 probability rows. The prefix replays at 0 and 32 also match exactly between builds. All four draft-3 generations accept 87/118 draft tokens, or 73.73%, over forty rounds.

Draft 0 and draft 3 choose the same output tokens, but their probability rows are not identical. The largest chosen-token log-probability difference is 0.12464; the largest shared top-32 difference is 1.42216. The corrected stock reference now reproduces the same differences exactly. This removes the evidence for an additional MoE-path numerical difference in these captured controls; it does not establish full-vocabulary logit equivalence, general acceptance invariance or equivalence for every input.

### Throughput and build provenance

Handoff `moe-decode-ab-20261102-091934-511712`, draft maximum 3, uses `55ee91f` against `45d51cb` for the uninstrumented `01-reference` phases. All four requests have 5121 prompt tokens, 128 completion tokens, content hash prefix `93595777`, and 74/156 accepted draft tokens (47.44%). These are comparable generation workloads within this run:

| State | Metric | Corrected reference | Candidate | Change |
| --- | --- | ---: | ---: | ---: |
| Cold | Prefill tokens/s | 351.05 | 498.96 | +42.14% |
| Warm | Prefill tokens/s | 456.12 | 579.40 | +27.03% |
| Cold | Decode tokens/s | 22.00 | 23.05 | +4.76% |
| Warm | Decode tokens/s | 34.93 | 36.06 | +3.22% |
| Cold | Request seconds | 20.3910 | 15.8057 | -22.49% |
| Warm | Request seconds | 14.8927 | 12.3901 | -16.80% |

Changes use the underlying timing precision; rounded console rates can differ slightly. These are single cold/warm observations, not confidence intervals or long-context measurements. The different acceptance percentage from the numerical fixture belongs to a different request, not a contradiction between builds in this A/B run.

The candidate `02-measure` and `03-profile` executables are still at `579880d`, before the cutoff and cropped-lookup service fixes. Their completions and draft counts differ from `01-reference`. The baseline uses the corrected `55ee91f` in all three phases. Therefore the candidate host byte counts, allocation peaks and GPU profile must not be presented as measurements of `45d51cb` or joined with its throughput as a current-build efficiency result. No profiler coverage warnings, unknown frees or unsupported allocation events are reported.

### Decode timeline evidence and next target

Raw profiler timestamps align with the HTTP monotonic request timestamps. Approximate decode windows use `monotonic_end_us - predicted_ms` through `monotonic_end_us`; the small HTTP/serialization tail is included. GPU durations below are interval unions per agent, not sums of overlapping activity. They describe the older candidate `579880d` and indicate where to inspect the current implementation:

| Older candidate activity | Cold decode, 8.279 s | Warm decode, 4.238 s |
| --- | ---: | ---: |
| XTX non-`batchMemOp` kernel union | 0.850 s | 0.963 s |
| XT non-`batchMemOp` kernel union | 0.931 s | 0.720 s |
| XTX `batchMemOp` wait union | 2.715 s | 0.00389 s |
| XT `batchMemOp` wait union | 1.301 s | 0.00279 s |
| Host `hipStreamSynchronize` union | 6.751 s / 10854 calls | 2.708 s / 10849 calls |

These are not additive wall-time components. Other wait helpers are included in the non-`batchMemOp` category; in particular, the cold XT window contains 0.218 s of `streamOpsWait`. Warm host-to-device copy unions are only 3.42 ms for the XTX and 4.76 ms for the XT; the XT-to-host copy union is 32.42 ms. The warm host records contain 2639 small-batch device lookups and no lookup misses. The corrected reference also has high host synchronization time: 2.782 s over 15690 calls in its 4.474 s warm decode window. Reducing the call count alone has not removed most of the waiting.

In the older candidate's warm HIP trace, 212 synchronizations immediately follow an event-record/event-wait pair and occupy 1.430 s; many such waits last about 24 ms. A second group immediately follows memcpy/synchronize and occupies 1.104 s. API adjacency is not a call stack or proof of causality. These timings include useful GPU execution and instrumentation overhead, and are not an achievable speedup estimate. They prioritize inspection of backend-copy/split barriers and DFlash input/state handoffs ahead of further warm-cache admission tuning or another small GEMM specialization. Cross-layer dependencies, copy completion, slot lifetime and buffer reuse still require ordering.

The older profile also records target compute reservations of 1176.75 -> 676.03 MiB on ROCm0, a 500.72 MiB difference, with ROCm1 unchanged at 646.23 MiB. The XTX process allocation peak differs by 445.25 MiB. These are old-build observations, not current physical-VRAM savings; allocation peaks include all HSA allocations and the two process peaks need not occur at corresponding execution points. Cache budgets remain unchanged.

### One focused current-candidate profile

No new native change is needed for the next capture. Rebuild only the candidate measurement executable using its existing ROCm cache, then collect only its `03-profile` phase. The test-tool-only `--candidate-profile-only` selection starts one fresh server with the same cold/warm fixture and archives the result; it requires no baseline or uninstrumented executable and emits no A/B throughput table. All candidate activation, acceptance, trace-completeness and profiler checks remain in place. Existing suite defaults are unchanged. Do not rerun the native correctness controls or the six-phase suite for this capture.

```fish
begin
    set -lx ROCM_PATH $HOME/opt/rocm-10.1
    set -lx HIP_PATH $ROCM_PATH
    set -lx LD_LIBRARY_PATH $ROCM_PATH/lib $LD_LIBRARY_PATH

    git pull --ff-only
    and cmake -S . -B build-rocm-moe-stream-measure -DLLAMA_MOE_MEASURE=ON
    and cmake --build build-rocm-moe-stream-measure --target llama-server ggml-hip -j28
    and python3 scripts/moe-measure/stream-ab.py --candidate-profile-only --draft-max 3
end
```

This capture can be compared with the existing corrected-reference profile for the same fixture. Its instrumented request timings remain diagnostic, not a replacement for uninstrumented throughput. A current trace must confirm the warm/cold bottlenecks before changing scheduler synchronization. No execution kernel, cache policy, graph, backend fallback or application runtime option is modified by this update.

Local validation: 25 focused Python cases pass (9 stream-suite, 3 decode-wrapper and 13 completion-diagnostic cases), including a mocked controller run that needs only the candidate measurement binary, starts only `03-profile` and preserves the handoff archive. The Python type check, EditorConfig check, CLI help check and `git diff --check` pass. No native rebuild or GPU/model run is needed for these tooling/documentation changes; the new physical capture remains open. The feature branch is not merged into production.

## 2026-11-02: cache-view fusions and repeated DFlash synchronization

The current candidate capture `moe-stream-ab-20261102-100257-793421` uses `f7cc734` and the same request bytes as the earlier A/B. Both requests produce content hash prefix `93595777` and accept 74/156 draft tokens. All 2874 layer groups per request are complete. Warm small-batch routing has 2639 device lookups, no misses and no expert H2D/D2D bytes. Cold small batches upload 13.5865 GiB of expert weights. Total expert H2D is 115.0606 GiB cold and 100.0683 GiB warm; expert D2D is zero. Target reservations are 676.03 MiB on ROCm0 and 646.23 MiB on ROCm1, retaining the observed 500.72 MiB ROCm0 reduction relative to the corrected reference. The XTX process HSA allocation peak is 445.25 MiB lower, not a direct measurement of driver VRAM.

The approximate warm decode window is 4.116 s. Its host stream-synchronization union is 2.688 s across 10810 calls; 1.643 s of those intervals overlap recorded GPU kernels/copies. The GPU kernel/copy union across both devices is 1.723 s. Device MoE `batchMemOp` unions are 4.02 ms and 2.79 ms. These quantities overlap and do not imply that 2.688 s can be removed. About 223170 kernel dispatches include 26596 Q8_1 quantizations. The host-submission intervals point primarily at target verify, but do not establish call-stack attribution or an achievable speedup.

### Implementation

On RDNA3 HIP, the backend now recognizes `MUL_MAT_ID -> weight VIEW -> MUL_MAT_ID -> GLU` with a shared input and compatible gate/up weights. The view must refer directly to an already allocated WEIGHTS bank with no computed producer. Moving this metadata-only view before the first matmul exposes the existing three-node gate/up/GLU fusion. No weight data is copied, and no new device kernel or quantization arithmetic is introduced. CUDA, Vulkan, dense matmuls and streaming-prefill operations do not enter this reorder.

Before allocation, eligible routed gate/up/GLU inputs are retained through the GLU output. This protects the activation, slot IDs and weight views from output aliasing when the existing fused MMVQ kernel runs. Eligibility for these allocation dependencies does not depend on query count, preserving the reserve/decode topology policy. Runtime dispatch keeps its existing type, batch, padding, graph-use and memory-overlap guards. Unsupported pairs, layouts and larger batches retain their existing dispatch. The actual reservation impact must still be checked on the physical model; longer input lifetimes can increase working memory.

`decode_device_inputs()` now calls the global target synchronization only when `n_queued_tokens` is nonzero. The counter is incremented before target execution and cleared only after scheduler streams, feature copies, MoE workers and performance accounting have finished. Target logits access normally performs that synchronization before DFlash injection. This removes the repeated drain after that completed work, while pending target work keeps the original synchronization and timing behavior. Existing feature-buffer reader events and overwrite protection remain. No additional producer-event system is needed for this case, and useful target execution time is not removed by this change.

The numerical backend fixture covers seven weight formats at widths 1 and 4, strided slot IDs, nonzero-offset bank views, a computed shared activation, width 8 fallback and different-input rejection. Direct backend comparisons pre-expand the weight views to exercise the existing fused arithmetic; they do not run scheduler graph optimization. Actual automatic hoisting and allocation lifetime behavior therefore require the model/scheduler run below. The existing DFlash device-input fixture now honors `--backend ROCm0` or `ROCm1`, using one sequence for the native GPU path; its default CPU two-sequence permutation test remains. It compares F16/Q8_0 KV state bytes, selector outputs, repeated buffer reuse, chunking and invalid-input handling.

### Focused physical validation

Use an incremental build and compare against the current working feature implementation, keeping the model and cache sizes fixed. Before replacing either candidate executable, save both complete `bin` directories once. The A/B launchers now put the selected executable's own directory first in `LD_LIBRARY_PATH`, so the saved baseline uses its saved shared libraries rather than the rebuilt ones. The existing ROCm library path remains available. Reuse the saved directories for repeated measurements of this change.

```fish
begin
    set -lx ROCM_PATH $HOME/opt/rocm-10.1
    set -lx HIP_PATH $ROCM_PATH
    set -lx LD_LIBRARY_PATH $ROCM_PATH/lib $LD_LIBRARY_PATH
    set -l before build-rocm-moe-before-verify-fusions

    if test -x $before/reference/llama-server; and test -x $before/measure/llama-server
        echo "Vorherige Builds werden wiederverwendet: $before"
    else
        test ! -e $before
        and mkdir $before
        and cp -a build-rocm-moe-stream-reference/bin $before/reference
        and cp -a build-rocm-moe-stream-measure/bin $before/measure
    end
    and git pull --ff-only
    and cmake -S . -B build-rocm-moe-stream-reference -DLLAMA_MOE_MEASURE=OFF -DLLAMA_BUILD_TESTS=ON
    and cmake --build build-rocm-moe-stream-reference --target llama-server test-backend-ops test-llama-archs -j28
    and cmake -S . -B build-rocm-moe-stream-measure -DLLAMA_MOE_MEASURE=ON
    and cmake --build build-rocm-moe-stream-measure --target llama-server ggml-hip -j28
    and ./build-rocm-moe-stream-reference/bin/test-backend-ops -b ROCm0 -o MUL_MAT_ID_CACHE_GLU
    and ./build-rocm-moe-stream-reference/bin/test-backend-ops -b ROCm1 -o MUL_MAT_ID_CACHE_GLU
    and ./build-rocm-moe-stream-reference/bin/test-llama-archs --test-dflash-device-inputs --backend ROCm0 -s 42
    and ./build-rocm-moe-stream-reference/bin/test-llama-archs --test-dflash-device-inputs --backend ROCm1 -s 42
    and python3 scripts/moe-measure/completion_diagnose.py --draft-max 0 \
        --baseline-reference build-rocm-moe-corrected-reference/bin/llama-server
    and python3 scripts/moe-measure/completion_diagnose.py --draft-max 3 \
        --baseline-reference build-rocm-moe-corrected-reference/bin/llama-server
    and python3 scripts/moe-measure/decode-ab.py --draft-widths 3 --profile \
        --baseline-reference $before/reference/llama-server \
        --baseline-measure $before/measure/llama-server
end
```

No old `test-moe-stream` sweep or extra draft-width sweep is required. The completion control still uses the corrected stock reference at `55ee91f`; the performance baseline is the saved, already tested MoE-feature build. Check tokens, probability rows, acceptance and reservations before attributing any change to performance. In the new GPU profile, eligible routed kernels should use `mul_mat_vec_q_moe<..., true>` and corresponding Q8_1/GLU launches should decline. A numerical backend test alone does not prove activation in the production graph or a throughput improvement. Remaining stream synchronization time must not be interpreted as wholly avoidable waiting.

Local validation: CPU Release builds of `test-backend-ops` and `test-llama-archs` pass. All 16 cache-view numerical cases pass on CPU, and the existing DFlash device/host fixture passes for F16 and Q8_0 KV, including state-byte and selector comparisons. The 26 focused Python cases pass (10 stream-suite, 3 decode-wrapper and 13 completion-diagnostic cases), as do the Python type check, EditorConfig and `git diff --check`. A temporary copy of the CPU `bin` directory resolves all five llama/ggml shared libraries from that copy with the new library-path prefix. The copy is removed after validation. No HIP/CUDA SDK or physical GPU is available here, so the HIP host code is not compiled locally, native fused arithmetic is not executed, and throughput, acceptance, graph reservation and peak allocation effects of this change remain open. No production-branch merge is made.

## 2026-11-02: batch scheduler input transfers

### Hardware results for the parent revision

The two `20261102-121606-804843` / `20261102-121830-683387` completion handoffs validate `2974c134` against the corrected reference at draft maxima 0 and 3. Tokens and complete recorded top-32 probability arrays match within and across builds, including prefix replays at 0 and 32. This does not measure full-vocabulary logits.

Handoff `moe-decode-ab-20261102-122034-456747` compares the preceding MoE feature against `2974c134`, not against stock execution. All requests have the same completion hash and 74/156 accepted drafts. The uninstrumented cold/warm decode rates are 22.30/34.90 versus 23.92/36.57 tokens/s; prefill is 500.24/576.03 versus 493.83/572.24 tokens/s. These single observations show +7.3%/+4.8% decode and -1.3%/-0.7% prefill, not confidence intervals or proof of regression-free performance. In the approximate warm profiler window, decode is slightly slower instead; profiler timing is not an uninstrumented speed verdict.

Fused routed kernels execute on both GPUs. Whole-session kernel dispatches decrease from 555757 to 539779 and Q8_1 quantizations from 55462 to 50136. Host H2D bytes and compute reservations are unchanged, with no material additional GPU allocation-peak reduction. The candidate's approximate warm decode window still contains 10706 `hipStreamSynchronize` calls. Their summed duration includes useful GPU execution and profiler overhead, and must not be treated as removable time.

### Scheduler change

For a scheduler with one copy of its compute storage, non-weight split inputs now share one destination drain. The allocator already places all split-input copies before that split's compute nodes; no computation is submitted between these copies. Repeating the destination drain for each independent input therefore protects the same preceding readers again. Existing asynchronous inter-backend copies retain their source/destination ordering, and a failed asynchronous copy still synchronizes its source before the synchronous fallback.

Host graph inputs up to 64 KiB use the backend's existing asynchronous setter when the destination has its default buffer type. Larger masks and embeddings retain synchronous copies, avoiding large stage-pool allocations and eviction of existing upload slots. One completion wait finishes all borrowed host reads before any expert-copy callback or scheduler return. Caller input storage can still be overwritten immediately after `graph_compute_async()` returns. On HIP this uses the existing malloc stage pool, including its existing settings and tensor filter; no new pool, allocation policy, runtime switch or input cache is added. Other buffer types and backends without an asynchronous setter retain synchronous copies. Three eligible HIP inputs replace three destination drains plus three synchronous buffer-copy waits with two backend drains; this call-count reduction is not a throughput prediction.

Multiple-copy schedulers retain the previous per-input path. Host expert weights and their callbacks retain the original copy helper. Graph construction, allocation lifetimes, reservations, conservative packed-mask coverage, arithmetic, routing, KV contents and cache policy are unchanged. A graph without non-weight split inputs adds no drain.

### Focused validation

`test-backend-sched` uses a CPU-backed deferred queue that borrows source pointers until synchronization. Eighteen cases cover setters present/absent, one/multiple scheduler copies, CPU-produced intermediates, host-weight callbacks, six graph reuses, caller input poisoning immediately after return, pending readers when the next call reuses input-copy storage, and the large-input fallback. The asynchronous single-copy case queues three host writes together and checks that only two backend synchronization calls precede compute submission. Its callback checks an actual preceding graph result. The optional `--backend ROCm0` / `ROCm1` mode runs nine small configurations at widths 1, 4 and 1024 on the selected real backend. With `--peer`, three more cases produce an input on the other GPU, covering asynchronous peer copies alongside host inputs and weight callbacks. This does not run the expensive MoE kernel sweep.

Keep the already validated `2974c134` binaries as the performance baseline. Save both complete `bin` directories before rebuilding. The launcher prefixes each selected binary's own directory in `LD_LIBRARY_PATH`, so the saved shared libraries remain part of that baseline.

```fish
begin
    set -lx ROCM_PATH $HOME/opt/rocm-10.1
    set -lx HIP_PATH $ROCM_PATH
    set -lx LD_LIBRARY_PATH $ROCM_PATH/lib $LD_LIBRARY_PATH
    set -l before build-rocm-moe-before-input-batching

    if test -x $before/reference/llama-server; and test -x $before/measure/llama-server
        echo "Vorherige Builds werden wiederverwendet: $before"
    else
        test ! -e $before
        and mkdir $before
        and cp -a build-rocm-moe-stream-reference/bin $before/reference
        and cp -a build-rocm-moe-stream-measure/bin $before/measure
    end
    and git pull --ff-only
    and cmake -S . -B build-rocm-moe-stream-reference -DLLAMA_MOE_MEASURE=OFF -DLLAMA_BUILD_TESTS=ON
    and cmake --build build-rocm-moe-stream-reference --target llama-server test-backend-sched -j28
    and cmake -S . -B build-rocm-moe-stream-measure -DLLAMA_MOE_MEASURE=ON
    and cmake --build build-rocm-moe-stream-measure --target llama-server ggml-hip -j28
    and ./build-rocm-moe-stream-reference/bin/test-backend-sched
    and ./build-rocm-moe-stream-reference/bin/test-backend-sched --backend ROCm0 --peer ROCm1
    and ./build-rocm-moe-stream-reference/bin/test-backend-sched --backend ROCm1 --peer ROCm0
    and python3 scripts/moe-measure/completion_diagnose.py --draft-max 0 \
        --baseline-reference $before/reference/llama-server
    and python3 scripts/moe-measure/completion_diagnose.py --draft-max 3 \
        --baseline-reference $before/reference/llama-server
    and python3 scripts/moe-measure/decode-ab.py --draft-widths 3 --profile \
        --baseline-reference $before/reference/llama-server \
        --baseline-measure $before/measure/llama-server
end
```

Local validation: CPU Release and Debug builds pass all 18 deferred-queue cases; the Debug run also enables AddressSanitizer and UBSan. LeakSanitizer cannot run under this environment's process tracing, so leak detection is disabled for that run. All nine real CPU-backend configurations pass. EditorConfig and `git diff --check` pass. No new Python or kernel code is introduced.

The HIP build, both physical backend fixtures, production logits/DFlash acceptance, Vulkan execution and throughput/peak-allocation measurements remain open. This environment has no HIP SDK or physical GPU. No production-branch merge is made.

### Test callback stream ordering

The first physical run passes all deferred cases, the four ROCm0 width-1 configurations and the first two width-4 configurations, then fails the width-4 multiple-copy callback assertion. That scheduler branch retains the old execution path. The test's direct buffer getter does not order its read after the backend's queued compute: a GPU event wait orders the GPU queue, not the callback's CPU read. The callback now queues `tensor_get_async` on the supplied backend and synchronizes that backend before checking the value, matching the existing production MoE callback's readback pattern. Production code and numerical tolerances are unchanged.

The deferred fixture now implements same-queue GPU events and queued reads. Its event wait preserves FIFO ordering without draining the CPU queue, while event synchronization can conservatively drain the queue. With these events the original callback reproduces the same assertion locally; after the readback correction all 18 cases pass again in Release and under ASan/UBSan (leak detection remains unavailable), as do the nine CPU configurations. Event-use assertions prevent silently falling back to the earlier no-event fixture.

Only `test-backend-sched` needs rebuilding for this correction, with `-j28`. Keep the saved `build-rocm-moe-before-input-batching` baseline and the already built server binaries. Rerun the deferred fixture and both `--backend ... --peer ...` GPU commands, then resume the two completion controls and draft-3 A/B command above. The corrected physical GPU tests and subsequent model/performance results remain open.

## 2026-11-02: finish small host uploads before peer dependencies

### Results of the input-batching round

The supplied console log passes all 18 deferred fixtures and all twelve configurations on each physical GPU, including peer copies in both directions. Completion handoffs `20261102-163129-895170` and `20261102-163359-542526` compare the saved input-batching baseline against the candidate at draft maxima 0 and 3. All repeated tokens, complete recorded top-32 probability arrays and prefix replays at 0 and 32 match exactly. The greedy draft-3 fixture accepts 87/118 drafts in both builds; the different performance request accepts 74/156 in all phases.

Handoff `moe-decode-ab-20261102-163548-631482` has native build IDs `2974c1347` and `22552b3a7` in both measurement phases and the uninstrumented runs. The repository HEAD `050641b` is the later test-only correction, not a different server execution path. The numerical handoffs record different `libggml-base` hashes and the expected saved-library path prefixes despite identical main-executable hashes.

| Uninstrumented metric | Baseline | Input batching |
| --- | ---: | ---: |
| Cold prefill tokens/s | 504.04 | 504.99 |
| Warm prefill tokens/s | 574.51 | 584.78 |
| Cold decode tokens/s | 22.97 | 23.56 |
| Warm decode tokens/s | 36.38 | 37.09 |
| Cold request seconds | 15.7226 | 15.5627 |
| Warm request seconds | 12.4346 | 12.2101 |

This is one A/B pair, not a stable-throughput or long-context guarantee. The profiled cold decode is slower instead. H2D bytes, compute reservations, GPU allocation peaks and all 539779 whole-session kernel dispatches are unchanged. The warm host trace has 2639 device lookups with no misses in both builds. Cache admission cannot remove more expert uploads from that captured warm-decode workload.

Approximate decode windows again use HTTP monotonic end minus native `predicted_ms`. Warm stream synchronizations decrease from 10706 to 4053, but their time union only decreases from 2652.65 to 2592.45 ms. Intervals overlapping recorded GPU activity account for 1632.90 and 1610.37 ms, respectively. The durations include useful GPU work and profiler overhead, not removable CPU overhead. Grouping by graph-begin and first-reservation timestamps places 1223.93 ms of candidate waits longer than 0.5 ms inside target graph submission and 999.62 ms after it. Draft submission and its subsequent interval contain 147.83 and 108.62 ms. These are graph regions, not captured call stacks.

### Transfer-order correction

The source confirms an avoidable ordering in the single-copy input helper. A peer copy can enqueue an event wait on the destination stream before small host-input uploads. Completing the borrowed host reads at the end of that group then also waits for the earlier peer dependency. The CPU cannot submit the split's GPU work until that wait completes. The trace's peer/event/upload sequences are consistent with this mechanism; they do not establish a speedup bound.

The helper now completes eligible small host-input uploads first, then submits all remaining non-weight inputs in their previous relative order. Only the existing asynchronous host-input path is moved: the 64-KiB limit, buffer-type checks, stage-pool settings, large-copy fallback, host expert callbacks and multiple-copy path are retained. The initial destination drain still protects previous readers. Borrowed host storage is fully read before caller return; existing device-copy events and source ordering protect peer data. No graph compute occurs between these input copies, and the allocator already places every split input before the split's compute nodes. No allocation plan, graph topology, arithmetic, cache policy or runtime switch changes. GPU calculation time remains necessary even when its CPU wait is moved out of graph submission.

The new deferred peer fixture snapshots on the producer queue after its compute and consumes the snapshot on the destination queue. It reproduces the old host-upload completion draining that peer dependency, then verifies that the corrected helper submits compute without such a drain. It also checks six graph reuses, previous-reader protection and immediate host-input poisoning without timing thresholds. There are now 19 default cases. With `--peer`, the real-backend fixture adds a peer case without a host-weight callback at each width, retaining the existing callback case: fifteen configurations per selected GPU.

Local validation passes all 19 cases in CPU Release and Debug with ASan/UBSan, plus all nine real CPU-backend configurations. Leak detection remains disabled because LeakSanitizer cannot run under this environment's process tracing. The old transfer order fails the new peer-drain assertion before this correction. EditorConfig, ASCII checks for changed C++ lines and `git diff --check` pass.

### Hardware check

Save the currently validated complete `bin` directories as a new baseline before rebuilding. Keep the earlier baseline folders. This round needs the short scheduler tests, one draft-3 completion diagnostic (including its non-speculative fixed-prefix replays), and the existing draft-3 A/B profile. It does not need a new MoE kernel or draft-width sweep.

```fish
begin
    set -lx ROCM_PATH $HOME/opt/rocm-10.1
    set -lx HIP_PATH $ROCM_PATH
    set -lx LD_LIBRARY_PATH $ROCM_PATH/lib $LD_LIBRARY_PATH
    set -l before build-rocm-moe-before-host-input-order

    if test -x $before/reference/llama-server; and test -x $before/measure/llama-server
        echo "Vorherige Builds werden wiederverwendet: $before"
    else
        test ! -e $before
        and mkdir $before
        and cp -a build-rocm-moe-stream-reference/bin $before/reference
        and cp -a build-rocm-moe-stream-measure/bin $before/measure
    end
    and git pull --ff-only
    and cmake -S . -B build-rocm-moe-stream-reference -DLLAMA_MOE_MEASURE=OFF -DLLAMA_BUILD_TESTS=ON
    and cmake --build build-rocm-moe-stream-reference --target llama-server test-backend-sched -j28
    and cmake -S . -B build-rocm-moe-stream-measure -DLLAMA_MOE_MEASURE=ON
    and cmake --build build-rocm-moe-stream-measure --target llama-server ggml-hip -j28
    and ./build-rocm-moe-stream-reference/bin/test-backend-sched
    and ./build-rocm-moe-stream-reference/bin/test-backend-sched --backend ROCm0 --peer ROCm1
    and ./build-rocm-moe-stream-reference/bin/test-backend-sched --backend ROCm1 --peer ROCm0
    and python3 scripts/moe-measure/completion_diagnose.py --draft-max 3 \
        --baseline-reference $before/reference/llama-server
    and python3 scripts/moe-measure/decode-ab.py --draft-widths 3 --profile \
        --baseline-reference $before/reference/llama-server \
        --baseline-measure $before/measure/llama-server
end
```

The new HIP build, physical peer tests, production numerical/acceptance comparison, Vulkan execution, end-to-end performance and allocation peaks remain hardware checks. No HIP SDK or physical GPU is available here. No production-branch merge is made.

## 2026-11-02: dense verify GLU and separate DFlash reserves

### Parent hardware results

The `20261102-173033-138119` completion handoff validates `b8b49e07` against `22552b3a`: tokens and all recorded probability rows match, including prefix replays. The corresponding `173230-321333` A/B has identical completion hashes and 74/156 accepted drafts. Uninstrumented cold decode is 23.31 -> 24.52 tokens/s; warm decode is 37.36 -> 37.60 tokens/s. The single warm observation (+0.6%) is not evidence of a stable speedup.

Warm target-submission intervals shrink from 1.444 s to 0.223 s, but much of the stream wait moves to result retrieval. Transfer bytes, kernel counts and GPU allocation peaks do not change. Removing more synchronization calls is not equivalent to removing the GPU work they wait for.

### Dense verify fusion

The initial RDNA3 HIP implementation enabled contiguous two-dimensional Q6_K/Q8_0 gate/up pairs without bias at widths 2-4. The hardware results below narrow this to Q6_K. Gate and up share one Q8_1 input quantization; the separate GLU launch and second input quantization disappear. Dot products, reductions and quantization use the existing implementations. One-token dispatch, routed MoE paths and unsupported/bias/batched layouts retain their previous behavior.

Allocation dependencies keep the weights and activation alive through the fused output. This policy is independent of query width so reserve and runtime plans agree; it can modestly increase the target's conservative reservation. There are no new runtime controls or caches.

### DFlash2 reservation

Single-sequence DFlash2 with selector metadata and layer splitting reserves host/device F1 ingestion and the F2 noise graph at the GGUF block size. F1 covers both the full ubatch and one fewer row, following the hardware results below. The device measure graph includes gathers, concatenations and conservative source storage on the FC device before the target's device inputs exist. The no-allocation fit pass takes the maximum per backend across these layouts. Large F1 batches keep the configured batch/ubatch sizes.

Recovery after F1 reserves only bounded F2; it does not repeat the F1 measurements. A noise batch larger than the current bound extends it before execution and retains the larger bound for later recoveries. Such nonstandard batches can grow the compute buffers and require additional free memory. Multi-sequence contexts, tensor splitting, and drafts without the selector keep the original reservation policy. The target reserve and its packed KQ mask protection are unchanged; bounded F2 still reserves its packed mask whenever required, including M-RoPE.

The old 1024 x 248320 F32 noise-logit tensor is nominally 970 MiB. Actual production savings require per-GPU measurement: F1 workspaces, small F2, allocator overlap and padding still need memory. The existing MoE cache is not enlarged automatically.

### Validation and physical run

Local CPU Release validation passes the 17 focused `MUL_MAT_VERIFY_GLU` cases, the DFlash device/host fixture, graph recovery, and existing DFlash vision tests. The reservation fixture uses a 4096-token vocabulary, M-RoPE and 32-row ingest capacity. Compute reservation is 164992 versus 589824 bytes with full-noise reserve. F1/F2 logits and serialized rollback state match exactly through widths 4, 8, 4; the larger bound is retained, normal F1/F2 does not grow the buffers, and per-buffer-type no-allocation estimates cover actual reservations. F16/Q8_0 device/host KV and selector comparisons and the two-sequence fallback remain covered. These CPU tests do not execute HIP arithmetic.

The DFlash fixture also passes in Debug with AddressSanitizer and UndefinedBehaviorSanitizer; leak detection is disabled because it cannot run under the local ptrace setup. Native device-input cases assert that alternating F1/F2 does not grow the initial per-buffer-type compute reserves. That assertion remains a physical GPU check.

No HIP SDK or physical GPU is available here. The HIP build, fused numerical results, full-model acceptance, throughput and GPU peaks remain open. Vulkan runtime validation is also unavailable. Compare against saved `b8b49e07` binaries with unchanged cache budgets. The existing suite records reservations and peak allocations per device; its uninstrumented run decides speed.

Save the current binaries before pulling, then run this in fish:

```fish
begin
    set -lx ROCM_PATH $HOME/opt/rocm-10.1
    set -lx HIP_PATH $ROCM_PATH
    set -lx LD_LIBRARY_PATH $ROCM_PATH/lib $LD_LIBRARY_PATH
    set -l before build-rocm-moe-before-verify-reserve

    if test -x $before/reference/llama-server; and test -x $before/measure/llama-server
        echo "Vorherige Builds werden wiederverwendet: $before"
    else
        test ! -e $before
        and mkdir $before
        and cp -a build-rocm-moe-stream-reference/bin $before/reference
        and cp -a build-rocm-moe-stream-measure/bin $before/measure
    end
    and git pull --ff-only
    and cmake -S . -B build-rocm-moe-stream-reference -DGGML_HIP=ON -DGPU_TARGETS='gfx1100;gfx1101' -DLLAMA_MOE_MEASURE=OFF -DLLAMA_BUILD_TESTS=ON
    and cmake --build build-rocm-moe-stream-reference --target llama-server test-backend-ops test-llama-archs -j28
    and cmake -S . -B build-rocm-moe-stream-measure -DGGML_HIP=ON -DGPU_TARGETS='gfx1100;gfx1101' -DLLAMA_MOE_MEASURE=ON
    and cmake --build build-rocm-moe-stream-measure --target llama-server ggml-hip -j28
    and ./build-rocm-moe-stream-reference/bin/test-backend-ops -b ROCm0 -o MUL_MAT_VERIFY_GLU
    and ./build-rocm-moe-stream-reference/bin/test-backend-ops -b ROCm1 -o MUL_MAT_VERIFY_GLU
    and ./build-rocm-moe-stream-reference/bin/test-llama-archs --test-dflash-device-inputs --backend ROCm0 -s 42
    and ./build-rocm-moe-stream-reference/bin/test-llama-archs --test-dflash-device-inputs --backend ROCm1 -s 42
    and python3 scripts/moe-measure/completion_diagnose.py --draft-max 0 \
        --baseline-reference $before/reference/llama-server
    and python3 scripts/moe-measure/completion_diagnose.py --draft-max 3 \
        --baseline-reference $before/reference/llama-server
    and python3 scripts/moe-measure/decode-ab.py --draft-widths 3 --profile \
        --baseline-reference $before/reference/llama-server \
        --baseline-measure $before/measure/llama-server
end
```

Eligible dense kernels should change to `mul_mat_vec_q<..., true, ...>` at widths 2-4, with fewer Q8_1 and GLU launches. Routed `mul_mat_vec_q_moe` is already fused and is not evidence for this change. Check draft and target reservations separately: the target can retain or slightly increase its reserve while the draft shrinks. The two implementation commits can be reviewed or reverted separately. No production-branch merge is made.

## 2026-11-02: physical results, Q8 rollback and partial F1 reserve

The `233917-924358` and `234140-876004` completion handoffs record matching tokens and complete top-32 probability arrays, including prefix replays at 0 and 32. The later launcher audit below shows that both sides actually loaded `a5cf563b`; these controls establish candidate repeatability, not a `b8b49e07`/`a5cf563b` probability comparison. The independently launched `moe-decode-ab-20261102-234333-737798` performance requests have identical hashes, 5121 prompt tokens, 128 completions and 74/156 accepted drafts.

Uninstrumented decode is 23.45 -> 23.60 tokens/s cold and 36.37 -> 37.39 warm. The instrumented warm controls are effectively unchanged (37.25 -> 37.30 and 31.72 -> 31.75); these are diagnostic controls, not independent production speed measurements. The 10.1% warm prefill gain in the uninstrumented pair is not repeated in the instrumented pair. A stable larger end-to-end speedup is not established.

The draft's measured compute high-water and total traced GPU allocation peaks both decrease. Target compute reservations and expert H2D/D2D bytes are unchanged; cache budgets remain 15200/9600 MiB.

| GPU | Draft compute before | Draft compute after | Reduction in traced GPU allocation peak |
| --- | ---: | ---: | ---: |
| RX 7900 XTX | 161.37 MiB | 18.10 MiB | 143.28 MiB |
| RX 7800 XT | 1041.38 MiB | 109.90 MiB | 931.48 MiB |

The sum of the per-device peak reductions is 1074.76 MiB (1.05 GiB). The tracer reports no coverage warnings or unknown frees; this is observed process allocation data, not an OS-wide VRAM reading. Draft host-compute reservation increases by about 50 MiB to cover host F1 fallback.

### Fusion decision from complete sections

Kernel-launch correlations identify matching warm decode sections, including input quantization and the GLU epilogue. The table sums GPU spans from the first quantization start to the last epilogue/fused-matmul end; these spans are not an end-to-end speedup bound.

| Section, width 4 | Count | Before | Fused |
| --- | ---: | ---: | ---: |
| Q6_K, RX 7900 XTX | 208 | 5.372 ms | 4.176 ms |
| Q6_K, RX 7800 XT | 104 | 3.622 ms | 2.462 ms |
| Q8_0 draft FFN, RX 7900 XTX | 208 | 19.459 ms | 23.925 ms |
| Q8_0 draft FFN, RX 7800 XT | 52 | 6.949 ms | 6.706 ms |

The Q6_K path is retained. Dense multi-token Q8_0 gate/up returns to its original dispatch: its dominant XTX workload regresses despite fewer launches, and the XT improvement is small. This keeps a simple type guard instead of a new per-shape/per-device tuning table. One-token and routed Q8_0 fusions are unaffected. Whole-session kernel counts fell by 3501 at `a5cf563b`; this alone did not establish a throughput gain. Performance of widths 2 and 3 remains unmeasured here.

### Nearly full F1 batches

The draft's XTX reserve stays at 18.10 MiB, but its XT reserve grows from 100.0039 to 109.8965 MiB at graph 7, a 1023-token F1 ingest. Reserving only the 1024-row layout missed that allocator placement. Initialization now measures both `n_ubatch` and `n_ubatch - 1` for host and device F1, merging per-backend maxima in the fit pass as well. F2 recovery still does not repeat F1 measurements. Global target and packed-mask reservation policy are unchanged.

The focused fixture now uses five feature streams, 1024-row capacity and F1 widths 1023, 1024 and 12 with F2 widths 4, 8 and 4, for F16 and Q8_0 KV. Native device-input coverage adds a 31-row batch split into 16 and 15 rows. It checks logits, serialized rollback state, retained larger-noise bounds, per-buffer-type fit estimates and normal-cycle allocation growth. The physical 2560-wide two-GPU allocation behavior still needs validation; CPU passing alone does not reproduce that GPU allocator boundary.

The updated fixture passes locally in CPU Release and Debug ASan/UBSan builds (LeakSanitizer disabled because of the container's ptrace restriction). Compact/full-noise CPU compute reservations are 15765504/22188032 bytes with F16 KV and 15847424/22925312 bytes with Q8_0 KV; these fixture sizes are not production VRAM measurements. Logits and serialized rollback states match exactly. EditorConfig and `git diff --check` pass. No HIP compiler or supported GPU is available locally.

Fixture correction on 2026-11-03: the initial 31-row GPU case incorrectly submitted all rows in one non-causal call with `n_ubatch=16`. The test now splits both device and host F1 calls at the ubatch boundary, matching the speculative caller. Its two-sequence CPU case uses 15 positions per sequence (30 rows), reproducing the same assertion before the fix and checking 16/14-row calls afterward; the single-sequence GPU case uses 16/15-row calls. Only the test and this document change; native GPU validation remains open.

The next hardware trial at `66867df` reaches the compute-buffer no-growth assertion. Its stack trace does not identify the buffer type, phase or byte increase, so the cause remains unresolved. The fixture now reports each new high-water beyond the initial bound after device F1 chunks, state readback, F2 noise and rollback, including backend, KV type, chunk width/offset and initial/previous/current bytes. It completes the remaining numerical checks and returns a failure code if any compute buffer exceeded the original bound. The bound is not relaxed. These reports exist only in the test binary; production execution and reservation are unchanged.

CPU coverage now includes both single-sequence optimized reservation and the existing two-sequence fallback. Both F16/Q8_0 cases pass in Release and Debug ASan/UBSan builds. The single-sequence CPU compute reservation remains unchanged through the tested cycles (246336 bytes for F16, 328256 for Q8_0). This does not reproduce the GPU growth. The next required input is the native `DFlash2 reserve growth:` report, not another end-to-end benchmark.

Re-use the saved `build-rocm-moe-before-verify-reserve` reference from `b8b49e07` and unchanged cache budgets. No new baseline copy or draft-width sweep is needed. Run this incremental fish block:

```fish
begin
    set -lx ROCM_PATH $HOME/opt/rocm-10.1
    set -lx HIP_PATH $ROCM_PATH
    set -lx LD_LIBRARY_PATH $ROCM_PATH/lib $LD_LIBRARY_PATH
    set -l before build-rocm-moe-before-verify-reserve

    test -x $before/reference/llama-server
    and test -x $before/measure/llama-server
    and git pull --ff-only
    and cmake -S . -B build-rocm-moe-stream-reference -DGGML_HIP=ON -DGPU_TARGETS='gfx1100;gfx1101' -DLLAMA_MOE_MEASURE=OFF -DLLAMA_BUILD_TESTS=ON
    and cmake --build build-rocm-moe-stream-reference --target llama-server test-llama-archs -j28
    and cmake -S . -B build-rocm-moe-stream-measure -DGGML_HIP=ON -DGPU_TARGETS='gfx1100;gfx1101' -DLLAMA_MOE_MEASURE=ON
    and cmake --build build-rocm-moe-stream-measure --target llama-server ggml-hip -j28
    and ./build-rocm-moe-stream-reference/bin/test-llama-archs --test-dflash-device-inputs --backend ROCm0 -s 42
    and ./build-rocm-moe-stream-reference/bin/test-llama-archs --test-dflash-device-inputs --backend ROCm1 -s 42
    and python3 scripts/moe-measure/completion_diagnose.py --draft-max 3 \
        --baseline-reference $before/reference/llama-server
    and python3 scripts/moe-measure/decode-ab.py --draft-widths 3 --profile \
        --baseline-reference $before/reference/llama-server \
        --baseline-measure $before/measure/llama-server
end
```

Check that the normal 1023-row ingest no longer grows the draft reserve, Q8_0 dense kernels return to the original path, and Q6_K fused sections remain active. Updated HIP compilation, native fixture results, production probabilities/acceptance and performance remain physical checks. No production-branch merge is made.

## 2026-11-03: external F1 source lifetime

The follow-up on gfx1100 and gfx1101 identifies the same increase on both cards, at the first 16-row device F1 chunk:

| KV | Initial GPU compute | First F1 peak | Increase |
| --- | ---: | ---: | ---: |
| F16 | 164480 bytes | 180864 bytes | 16384 bytes |
| Q8_0 | 246400 bytes | 262784 bytes | 16384 bytes |

All native K/V, selector and graph-transition numerical checks pass. The 1023/1024-row reservation fixture also passes on both GPUs. This failure is at a full F1 chunk, not a new query-width or partial-batch boundary; it does not justify another shape threshold or a fixed 16-KiB margin.

The original device F1 measure graph allocates its synthetic source tensors inside the compute allocator. Runtime source tensors already occupy external target staging. These cases have different source lifetimes and allocator reuse, so the first layout is not a guaranteed upper bound for the second. The reserve now measures both layouts for the full and nearly full F1 widths. The external layout uses a zero-byte backend buffer descriptor owned by the graph input, following ggml's existing no-allocation planning convention. It creates no source payload or VRAM allocation, and the graph is never executed.

The original owned-source measure remains in the per-backend maximum, preserving its conservative source-storage budget for fit. F2 recovery still measures only bounded noise; global packed KQ-mask reservation, runtime attention arithmetic and cache policy are unchanged. There are no new user controls or caches.

The focused fixture additionally verifies that no-allocation fit estimates cover separate owned/external F1 measurements for every backend buffer type. CPU Release and Debug ASan/UBSan pass for F16/Q8_0, single and multiple sequences, full/partial F1 ingestion, logits, selector lattice and serialized rollback (LeakSanitizer disabled because of the container's ptrace restriction). Its CPU compute sizes are unchanged.

The native follow-up at `fa5789c` passes on both gfx1100 and gfx1101 with F16 and Q8_0 KV. There are no reserve-growth or reserve-failure reports through the 16/15-row F1 chunks, subsequent 1/4-row inputs, F2 noise and rollback. K/V and selector comparisons pass, as do the full/partial F1 logits and fit checks.

The synthetic 1024-row fixture reports the following aggregate compute reservations (all backend buffer types summed, not production VRAM):

| KV | Compact F1/F2 reserve | Full-noise reference |
| --- | ---: | ---: |
| F16 | 16846848 bytes (16.07 MiB) | 30613504 bytes (29.20 MiB) |
| Q8_0 | 17010688 bytes (16.22 MiB) | 31432704 bytes (29.98 MiB) |

Both compact figures increase by exactly 1 MiB versus the previous hardware trial, matching one 256-wide F32 feature buffer at 1024 rows. The 16-row failure was the corresponding 16-KiB difference. A subsequent production A/B run is still needed to measure process peaks and throughput with this correction; no speedup is claimed for changing reservation alone. Keep the saved `b8b49e07` baseline, draft maximum 3 and unchanged cache budgets for that comparison.

## 2026-11-03: production validation and numerical-launcher correction

Handoff `moe-decode-ab-20261103-011409-070969` uses the saved pre-reservation baseline and `fa5789c`, with unchanged 15200/9600-MiB cache budgets and the same 5121-token prompt/128-token completion. All twelve performance requests have the same completion hash and accept 74/156 drafts. The launchers prefix each selected binary's own directory in `LD_LIBRARY_PATH`. The raw traces independently confirm different implementations: baseline Q6_K is unfused with the old draft reservations, whereas the candidate uses fused Q6_K, unfused Q8_0 and compact draft reservations.

| Uninstrumented metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Cold prefill tokens/s | 510.57 | 506.17 | -0.86% |
| Warm prefill tokens/s | 582.98 | 587.90 | +0.84% |
| Cold decode tokens/s | 23.40 | 23.61 | +0.87% |
| Warm decode tokens/s | 37.38 | 37.24 | -0.37% |
| Cold request seconds | 15.4859 | 15.5262 | +0.26% |
| Warm request seconds | 12.2102 | 12.1500 | -0.49% |

This pair establishes no stable throughput gain or material regression. Instrumented controls have different overhead and must not be averaged with the uninstrumented pair as additional production repetitions.

| GPU | Baseline draft compute | Candidate draft compute | Reduction in process GPU allocation peak |
| --- | ---: | ---: | ---: |
| RX 7900 XTX | 161.37 MiB | 18.10 MiB | 143.28 MiB |
| RX 7800 XT | 1041.38 MiB | 110.00 MiB | 931.37 MiB |

The sum of per-device peak reductions is 1074.65 MiB (1.05 GiB). Allocation tables have no coverage warnings, unknown frees or unsupported events. Draft compute is constant through both requests; the previous late increase is gone. Target compute high-water remains 676.03/646.23 MiB. The draft retains about 50 MiB more host-compute reserve; traced host allocation peak is 41.60 MiB higher. These are process allocation measurements, not OS-wide VRAM readings. Graph-plan reallocation messages still occur; they do not imply compute-buffer byte growth.

### Remaining decode cost

The warm host trace has 2639 small target MoE layer batches, all cache hits with zero expert H2D/D2D bytes, for both builds. Cache admission cannot remove uploads from this captured warm workload. Cold small batches upload 7330.27 MiB on ROCm0 and 6582.28 MiB on ROCm1, so cache coverage and prefill-to-decode preparation remain relevant for cold and changing workloads.

Complete warm Q6_K gate/up/GLU section spans fall from 9.013 to 6.630 ms over 312 sections. Q8_0 draft FFN spans return to baseline (26.320 versus 26.292 ms). The Q6_K saving is about 2.38 ms across 128 generated tokens, too small to explain a large end-to-end gain. Whole-session kernel dispatches fall from 539779 to 537853.

The approximate warm decode window is 3996.51 ms. The candidate executes 187528 target kernels and 20020 draft-generation kernels; 20713 target launches quantize Q8_1 activations. The target MoE kernels include IQ2_S, IQ4_NL and IQ3_XXS: the model's package label must not be treated as one uniform kernel quantization. The next kernel investigation should inspect common-input projection/quantization and complete HC/projection subgraphs, rather than expect another isolated shared-FFN fusion to change throughput substantially. This is an investigation direction, not a demonstrated speedup or a new implementation.

There are 3952 `hipStreamSynchronize` calls in that warm window, with a 2528.94-ms time union. Recorded GPU kernel/copy activity has a 1693.70-ms union across both cards. These intervals overlap, and the API waits include GPU execution and profiler overhead; neither the synchronization duration nor the remaining trace gaps are a removable-overhead estimate. The trace has no call stacks. Earlier input-order changes already moved much of the target wait from submission to output retrieval. Removing more calls without tracing the actual dependency would not establish more parallelism.

### Completion diagnostic library path

The separate `moe-completion-diagnose-20261103-011156-705915` handoff has identical tokens and recorded top-32 arrays, but both `/props` responses identify `b12054-fa5789c93`, and both traces reserve 18.10/110.00 MiB for the draft. Its baseline launcher selected the saved executable without prefixing the saved library directory. The executable's original library search path therefore selected the current build's libraries. Disk checksums of the files beside the saved executable did not establish which libraries were loaded.

This handoff validates repeatability of the candidate, not cross-build probability equivalence. The same issue is confirmed in the `233917` and `234140` controls, where both sides report `a5cf563b`. Cross-build probability conclusions from other archived-executable controls remain provisional until their library origin is checked. The performance A/B launchers already set the correct prefix and are unaffected; their matching generation hashes and DFlash counts remain valid evidence.

`completion_diagnose.py` now prefixes the selected binary's own directory immediately before invocation, using the same quoting and inherited-path handling as the performance launcher. A regression test with a saved path containing spaces and an inherited current-build path fails before this correction. All 14 completion-diagnostic unit tests, `ty 0.0.88` checks and EditorConfig checks pass afterward. This changes measurement tooling only; the server requires no rebuild.

One corrected draft-3 completion control is needed, using the existing binaries. Another performance A/B or GPU kernel sweep is not needed for the launcher fix:

```fish
git pull --ff-only
and python3 scripts/moe-measure/completion_diagnose.py --draft-max 3 \
    --baseline-reference build-rocm-moe-before-verify-reserve/reference/llama-server
```

The expected reference build is `b8b49e07`, with its old 161.37/1041.38-MiB draft reservation. The corrected cross-build top-32 result remains open. No production-branch merge is made.

## 2026-11-03: HC down-projection candidate for target verify

The next change targets the BF16 HC down-projection, not expert transfers. In the latest warm trace all small MoE batches hit the cache. The interval between `publish_lookup` and `finish_lookup` totals 15.45 ms on ROCm0 and 12.54 ms on ROCm1 over 2544 lookups. This includes graph scheduling, not just the memory wait. It is too small to treat the lookup wait as the main remaining bottleneck.

The HC down-projection has K=10240, M=320 and four queries. Its existing MMF launch has only ten blocks, with eight waves in each block. There are 3120 such launches on ROCm0 (92.35 ms) and 1924 on ROCm1 (52.21 ms). The following scale/SiLU kernels add work and graph nodes. These measured costs motivate a candidate; they do not establish its speedup.

### Kernel and dispatch

The candidate assigns each of the old eight warp partitions to its own block, giving 80 blocks for the production shape. It reuses the existing BF16 WMMA loop. Each partition keeps the same K start, K stride and instruction order. A second kernel sums partitions 0 through 7 in the original order, then applies the existing scale/bias expression and SiLU helper. There are two launches instead of the previous matmul, scale and SiLU launches.

Activation conversion, weight types and attention arithmetic are unchanged. The graph retains its original GGML operations; the backend recognizes the adjacent `MUL_MAT -> SCALE -> SILU` chain. Existing subgraph-output and memory-overlap checks protect fusion. Allocation dependencies keep its inputs alive through the last node and are batch independent, as for the existing GLU fusions. No graph reordering, new runtime option or cache is introduced.

Automatic dispatch is limited to RDNA3 wave32, dense contiguous BF16 weights/F32 activations and output, 4-8 queries, 32-512 output rows in multiples of 32, and K=8192-65536 in multiples of 512. The model's K=10240/M=320 case fits. Other types, shapes, backends, hints and observable intermediate outputs keep their existing path. Plain matmuls and widths 1-3 keep their original dispatch. Prefill is not switched to split K. QSA, KV storage, MoE policy, DFlash inputs and checkpoint handling are not changed.

The partial payload is 40 KiB for K=10240/M=320/four queries and at most 128 KiB per eligible operation, using the existing temporary backend pool. Pool allocation rounding and graph allocation dependencies still need checking in production peaks. The previous 1.05-GiB draft-reservation saving is not a measured result for this new candidate yet.

### Validation status

Local CPU CMake configuration/build and the 19 deferred scheduler cases pass. A host index-coverage check verifies unique partial writes for supported row/query boundaries, and checks that the old dispatcher chooses eight warps for every supported K. `git diff --check` and EditorConfig checks pass. No HIP compiler is installed here: configuration stops at `Failed to find HIP root directory`. The new HIP source has not been compiled or run here; GPU numerical equivalence, performance and memory peaks remain open. No inference speedup is claimed.

`test-hip-mmf-thin` checks the production kernel against the existing eight-warp MMF kernel. It also evaluates the actual GGML chain through the scheduler/allocator. It requires bitwise equality, without raising tolerances. Cases cover K=8192/10240/16384/65536, M=32/320/512, widths 4-8, plain and fused results, changing/zero/tiny inputs, poisoned scratch, tail guards, nonzero bias and a retained intermediate that must prevent fusion. Only the fused chain is enabled in inference; the plain split-K comparison is diagnostic.

Its optional `--bench` reports both variants with 64 operations per captured HIP graph and ten interleaved AB/BA samples. Each `split_us` includes the reduction and fused activation when `silu=1`. Median timings avoid timing host submissions of isolated tiny kernels. Reused weights can stay in cache, so the existing full-model A/B remains necessary. The test option does not alter server dispatch.

Save the current working binaries before rebuilding. Use a new snapshot directory to isolate this candidate from the earlier draft-reservation work. The previous saved references remain available for older comparisons. Run this fish block with the usual model/request templates:

```fish
begin
    set -lx ROCM_PATH $HOME/opt/rocm-10.1
    set -lx HIP_PATH $ROCM_PATH
    set -lx LD_LIBRARY_PATH $ROCM_PATH/lib $LD_LIBRARY_PATH
    set -l before build-rocm-moe-before-hc-thin

    if test -x $before/reference/llama-server; and test -x $before/measure/llama-server
        echo "Vorherige Builds werden wiederverwendet: $before"
    else
        test ! -e $before
        and mkdir $before
        and cp -a build-rocm-moe-stream-reference/bin $before/reference
        and cp -a build-rocm-moe-stream-measure/bin $before/measure
    end
    and git pull --ff-only
    and cmake -S . -B build-rocm-moe-stream-reference -DGGML_HIP=ON -DGPU_TARGETS='gfx1100;gfx1101' -DLLAMA_MOE_MEASURE=OFF -DLLAMA_BUILD_TESTS=ON
    and cmake --build build-rocm-moe-stream-reference --target llama-server test-hip-mmf-thin -j28
    and cmake -S . -B build-rocm-moe-stream-measure -DGGML_HIP=ON -DGPU_TARGETS='gfx1100;gfx1101' -DLLAMA_MOE_MEASURE=ON
    and cmake --build build-rocm-moe-stream-measure --target llama-server -j28
    and ./build-rocm-moe-stream-reference/bin/test-hip-mmf-thin --backend ROCm0 --bench
    and ./build-rocm-moe-stream-reference/bin/test-hip-mmf-thin --backend ROCm1 --bench
    and python3 scripts/moe-measure/completion_diagnose.py --draft-max 0 \
        --baseline-reference $before/reference/llama-server
    and python3 scripts/moe-measure/completion_diagnose.py --draft-max 3 \
        --baseline-reference $before/reference/llama-server
    and python3 scripts/moe-measure/decode-ab.py --draft-widths 3 --profile \
        --baseline-reference $before/reference/llama-server \
        --baseline-measure $before/measure/llama-server
end
```

The corrected completion launcher loads each binary's own libraries. Check `/props` build IDs as well as hashes, recorded probabilities and DFlash counts. In the candidate GPU trace the eligible HC down chain should use `mul_mat_f<..., false, true>` plus `mmf_thin_reduce_f32`; the corresponding separate scale/SiLU launches should disappear. Compare uninstrumented decode and request time, target/draft reservations, per-device peaks and unchanged acceptance before accepting a performance claim. A slower fused production shape needs dispatch adjustment or removal, not a looser correctness threshold. The branch remains for review; no merge to `codex-compat` is made.

### Native test scheduler correction

Both HIP builds completed on the user's ROCm 10.1 installation. The first test then stopped before launching a kernel: its scheduler had only the selected GPU, whereas `ggml_backend_sched_new()` requires the CPU backend last. This was a test fixture error, not a numerical result.

The fixture now keeps one CPU fallback alive through all cases and uses `{selected GPU, CPU}`. It explicitly assigns the input and all graph nodes to the selected GPU, checks its operator support, and verifies placement after allocation. The fallback cannot silently turn this GPU comparison into a CPU test. Kernels, dispatch, tolerances and benchmark arithmetic are unchanged.

Local checks pass: a smoke test extracted from the fixture setup runs twelve CPU-backed variants with two computes each and checks the selected backend placement; all nineteen existing deferred scheduler tests also pass. The smoke test does not validate HIP execution. GPU numerical and performance results remain open.

The saved `build-rocm-moe-before-hc-thin` baseline remains valid. Rebuild only the test target and resume the interrupted sequence in fish, using the existing ROCm environment:

```fish
begin
    set -l before build-rocm-moe-before-hc-thin
    git pull --ff-only
    and cmake --build build-rocm-moe-stream-reference --target test-hip-mmf-thin -j28
    and ./build-rocm-moe-stream-reference/bin/test-hip-mmf-thin --backend ROCm0 --bench
    and ./build-rocm-moe-stream-reference/bin/test-hip-mmf-thin --backend ROCm1 --bench
    and python3 scripts/moe-measure/completion_diagnose.py --draft-max 0 \
        --baseline-reference $before/reference/llama-server
    and python3 scripts/moe-measure/completion_diagnose.py --draft-max 3 \
        --baseline-reference $before/reference/llama-server
    and python3 scripts/moe-measure/decode-ab.py --draft-widths 3 --profile \
        --baseline-reference $before/reference/llama-server \
        --baseline-measure $before/measure/llama-server
end
```
