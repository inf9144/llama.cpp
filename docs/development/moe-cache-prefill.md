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
