# Compute graph reallocation diagnostics

This branch combines the production fix from `fix/graph-reallocation-recovery` (`fd1e65ec405cbfbedec4ecb2d52c7ef235acfdff`) with opt-in diagnostics. The fix restores the worst-case text allocation plan after an incompatible graph transition using the existing scheduler and buffers. The additional instrumentation only observes allocation and recovery; it adds no synchronization. See [text allocation recovery](graph-allocation-recovery.md) for the implementation and GPU validation checklist. Merge the clean fix branch independently when production validation is complete.

## Run

In the existing Bash startup script, add this before the `llama-server` command:

```bash
export GGML_SCHED_TRACE_REALLOC=1
```

Keep the current server arguments, including the current log verbosity. The diagnostic records use INFO, so `--log-verbosity 4` (or higher) is required when using the common server logger. The existing per-event `sched_graph_realloc` messages are unchanged.

Alternatively, from fish, run the existing script with a process-local setting:

```fish
env GGML_SCHED_TRACE_REALLOC=1 bash ./run_llama_swift-1.5-qwen3.8-27b.sh 2>&1 | tee graph-realloc-level1.log
```

Use the actual path of the startup script. No server argument needs to change.

| Setting | Behavior |
| --- | --- |
| Unset or `0` | No diagnostic objects, clocks, graph fingerprints, or diagnostic logging. |
| `1` | Exact allocation rejection reason, stage timings, batch metadata, bounded details and aggregates. |
| `2` | Also fingerprint every scheduled graph's tensor layouts and edges; compare against the previous scheduling and last reservation. |
| `3` | Level 2 plus details for every reallocation, recovery and backend buffer allocation attempt. Use only for a short capture. |

The variable is read once when a scheduler is created. Restart the server to change it. `GGML_SCHED_DEBUG_REALLOC` is a separate, existing abort-on-reallocation control; it is not needed for these diagnostics.

## Investigation

The original diagnostic commit was based on `codex-compat` at `ee25617c778fa1f88c002beba5b24ce9d75cfc90`; the recovery merge also includes `codex-compat` through `4bff787f7eb99c691ead79ea4730ae30761ddeac`. All four relevant upstream changes are ancestors:

| PR | Commit | Relevant behavior |
| --- | --- | --- |
| [#29634](https://github.com/ggml-org/llama.cpp/pull/29634) | `8019dc563` | Pipeline input copies include all input leaves, including inputs unused by a particular batch. |
| [#29958](https://github.com/ggml-org/llama.cpp/pull/29958) | `c173a53bd` | Stabilizes k-pool graph composition in the affected model builders. It is not a general allocator fix. |
| [#30020](https://github.com/ggml-org/llama.cpp/pull/30020) | `1a3011cc0` | Requests a fresh scheduler reservation when nextn extraction flags change. |
| [#30329](https://github.com/ggml-org/llama.cpp/pull/30329) | `972b5fcc1` | Adds the existing reallocation callback and same-size/changed-size counters. |

The existing graph-size comparison uses a scalar derived from node/leaf counts and scheduler-inserted inputs/dependencies. `changed=0` does not compare tensor dimensions, strides, operations, edges, or required allocation sizes. `n_same` and `n_diff` also exclude events with changed backend assignments.

`ggml_gallocr_node_needs_realloc()` has a misleading name: it returns true when the stored tensor allocation is valid. The diagnostic code preserves that contract. Its existing failure conditions are:

- A tensor now needs allocator-owned storage, but its previous allocation has `buffer_id < 0` (the previous tensor was external, unused, or a view).
- The current buffer-type-specific allocation size is greater than the stored `size_max` at that node/source position.

The outer check first rejects node-count and leaf-count changes, then checks node destinations and their sources. It does not separately validate every leaf allocation. This branch does not add such a check. A failing source that is also a leaf reports both indices.

The scheduler separately re-reserves when the first changed backend assignment also changes buffer type. The diagnostic records that exact decision, including the previous/current backend IDs and the node or leaf index; it does not run the allocator check that the existing short circuit skips.

`ggml_gallocr_reserve_n_impl()` rebuilds the memory plan, but only frees/allocates backend buffers if a buffer is missing or a required chunk grows. Replanning and backend allocation are therefore different events. Single-buffer allocators can replan internally and return success; those events are included with `automatic=1` even though they do not invoke the existing scheduler reallocation callback.

The production Qwen/MTP log identified loss of the initial worst-case text plan after switching to image embeddings and back. Runtime reservation replaces the stored tensor limits with those of the current batch. Alternating zero-output catch-up and single-output drafting then invalidate each other's `out_ids` and token-input capacities without growing the backend buffers. Model-free tests reproduce the oscillation and its removal after re-reservation. The complete attribution of later main-context size failures and the real GPU performance benefit still require measurement.

## Identity and graph metadata

Every enabled scheduler receives a monotonic process-local `sched` ID. `llama_context::sched_new()` attaches a stable `context` ID that survives scheduler recreation during that context's lifetime. The role is derived from the actual context parameters:

- `mtp_draft`: `LLAMA_CONTEXT_TYPE_MTP`, including MTP against the same model object as the target.
- `other_draft`: another context type with `ctx_other` set.
- `main`: a context without either property.
- `unassigned`: a scheduler used without a llama context.

The common MTP initializer explicitly sets `LLAMA_CONTEXT_TYPE_MTP`. `common_speculative_impl_draft_mtp::process()` catches up the draft context with no output flags; `draft()` runs output-producing draft decodes, and the target verifies the proposed batch. Metadata records `tokens`, `outputs`, `sequences`, first/last position in the first position plane, and `nextn_layer`. The last field is the selected MTP head, not the draft iteration for a single-head Qwen model.

The existing 88-node and 4512-node messages cannot by themselves identify a context. Match the new records' role, scheduler ID, node/leaf counts, and split count to verify the attribution on the real model. This branch has not executed that model on the two AMD GPUs.

Batch metadata comes from `process_ubatch` and explicit reservation calls. Multi-token batches may be either prompt processing or speculative verification. The low-level context does not know the server's exact request phase, requested speculative length, or which proposer won an MTP/N-Gram arbitration. Correlate those with existing speculative/server timing records; do not label every multi-token graph as prompt processing.

## Records

All new records start with `sched_realloc_diag:` and use space-separated `key=value` fields. Tensor/device names are quoted and escaped. Arrays contain no spaces. No tensor data, token IDs, prompt text, or token text are logged.

Records include:

- `event=scheduler`, `event=device`, `event=context`: identity, actual devices/buffer types, requested pipeline parallelism, and copy count.
- `event=realloc`: first three events per reason per scheduler, or every event at level 3; graph/batch characteristics and per-event timings. `success=0` includes `failure_stage=reserve`, `binding`, or `debug_abort` when the separate existing abort-on-reallocation control is enabled.
- `event=tensor`: the first failed tensor's name/op/type/layout, node/leaf/source index, old allocation limit, current required size, previous allocation buffer ID and current assigned backend/buffer type. `required_from_check=1` means the size came directly from the failing allocator comparison. For a missing previous allocation, the current assignment is measured only for explanation (`required_from_check=0`); that measurement does not decide validity.
- `event=buffer`: first three actual backend buffer allocation attempts per backend and scope (explicit reservation or reallocation; all at level 3), including chunk, replacement flag, old/requested/actual bytes, allocation-call time (`alloc_us`) and success.
- `event=summary`, `event=realloc_totals`: lifetime and reallocation-only aggregates, emitted at most once per five seconds of allocation activity, plus scheduler destruction. Graceful shutdown is needed for final summaries.
- `event=recovery`: first three recovery attempts per scheduler (every attempt at level 3, all failures), including total time, synchronization, planning, buffer allocation deltas and before/after compute-buffer sizes by backend ID. `reserve_tokens` and `reserve_outputs` describe the synthetic reservation, not an executed batch.
- `event=recovery_summary`: cumulative recovery-only costs and counts, alongside periodic/final summaries after the first recovery.

Example from a CPU smoke test with a same-count source-size increase (the production context/device labels will differ):

```text
sched_realloc_diag: event=context sched=1 context=1 role=synthetic
sched_realloc_diag: event=buffer sched=1 context=1 role=synthetic scope=realloc buffer=0 chunk=0 backend="CPU" buft="CPU" replacement=1 old_bytes=96 requested_bytes=160 actual_bytes=160 success=1 alloc_us=0
sched_realloc_diag: event=realloc sched=1 context=1 role=synthetic ordinal=1 reason=tensor_size success=1 failure_stage=none automatic=0 phase=smoke tokens=32 outputs=0 sequences=1 pos_first=1 pos_last=32 nextn_layer=0 nodes=1 leafs=1 prev_nodes=1 prev_leafs=1 reserved_nodes=1 reserved_leafs=1 splits=1 copies=1 copy=0 graph_size=1 prev_graph_size=1 backend_index=-1 backend_leaf=0 backend_old=-1 backend_new=-1 layout_changed=1 topology_changed=0 reserved_layout_changed=1 reserved_topology_changed=0 layout=7772526028952870449 prev_layout=4079136524388879649 reserved_layout=4079136524388879649 topology=8762066661691639763 prev_topology=8762066661691639763 reserved_topology=8762066661691639763 path_us=7 validation_us=0 sync_us=0 planning_us=1 buffer_us=1 binding_us=2 buffer_allocs=1 gpu_buffer_allocs=0
sched_realloc_diag: event=tensor sched=1 context=1 ordinal=1 node=0 leaf=0 src=0 name="synthetic_input" op=NONE type=f32 ne=[32,1,1,1] nb=[4,128,128,128] reserved_bytes=64 required_bytes=128 required_from_check=1 reserved_buffer=1 assigned_backend=1 backend="CPU" buft="CPU"
sched_realloc_diag: event=realloc_totals sched=1 context=1 validation_us=0 sync_us=0 planning_us=1 buffer_us=1 binding_us=2 buffer_allocs=1 gpu_buffer_allocs=0
```

`layout_changed`/`topology_changed` are `-1` when not measured. They describe differences, not additional causes for the allocator decision. Level 2 also prints hashes for the previous graph and last reservation, and corresponding `reserved_*_changed` fields. The fingerprints cover types, dimensions, strides, flags, view offsets, operations and indexed edges; they exclude pointers, data and operation parameters. A matching fingerprint is not proof that graphs are identical. These are not a replacement allocation-validity check.

## Timing and buffer accounting

All times are host-side microseconds from the existing monotonic `ggml_time_us()` clock (sub-microsecond stages can read as zero):

| Field | Scope |
| --- | --- |
| `validation_us` | Existing allocator validity check, including first and post-reserve checks. Does not include the scheduler's backend-assignment scan. |
| `sync_us` | Only the synchronization loop already present in the scheduler's reallocation fallback. Includes waiting for already pending GPU work. |
| `planning_us` | Reservation hash/allocator setup and memory planning, before backend buffer replacement. |
| `buffer_us` | Freeing/replacing backend buffers, including chunk allocation and buffer initialization; excludes the buffer-observer callback. |
| `binding_us` | Final buffer resets and tensor address binding after validation. |
| `path_us`, `max_event_us` | Entire reallocation allocation-path wall time, including scheduler checks, existing callback and synchronization; ends before printing that event's details. |
| `observer_us` | Graph fingerprint/metadata bookkeeping and diagnostic reporting measured by the observer. It is not an exact measurement of all instrumentation overhead. |

`event=summary` allocator timings/counts include explicit startup reservations and normal successful graph allocations. `event=realloc_totals` includes only the calls that needed a replan. Do not sum lifetime and reallocation-only totals. `path_us` already contains synchronization, planning and binding; do not add them to it. Some observer cost overlaps `path_us`; logging/clock reads and instrumentation also perturb the measurement.

The existing reallocation synchronization may simply wait earlier for work that would otherwise be waited for later. It is not automatically an equal increase in end-to-end latency. This branch adds no synchronization. Compare session throughput as well as these timings.

Buffer counters count successful calls to the backend buffer allocator, once per physical backend buffer chunk, with separate failure counts. They do not count merely changing a memory plan or aliases of the same buffer type. Startup buffers are included in lifetime totals; the reallocation-only totals and per-event deltas exclude them. GPU counts classify GPU/IGPU devices; CPU buffers are counted separately by subtraction. Backend-internal pooling, suballocation or driver residency changes are not visible here: a backend buffer allocation is not a proven driver-level `vkAllocateMemory`/HIP allocation. Use a backend/driver profiler if that distinction becomes material.

`buffer_growths` includes initial allocations with nonzero size (old size zero); `buffer_replacements` requires an old chunk at that index. Size-only probes can release buffers under existing allocator behavior; no backend allocation is counted for a probe. Partial allocation failures count successful chunk calls before the failure, even if subsequently cleaned up.

Counters belong to scheduler instances. Recreated schedulers emit separate final summaries with the same llama context ID. Sum only the final record for each scheduler when comparing lifetime totals. Periodic records are cumulative snapshots, not intervals. A reused compute graph may perform no allocation call, so periodic reporting can pause during reuse. Abrupt termination/OOM cannot produce a final summary.

Recovery records use separate scopes from `realloc_totals`. Their `recovery_us` includes the initial existing scheduler synchronization, full-memory simulation, construction of the measure graph and explicit worst-case reservation. `sync_us` includes all calls to the existing scheduler synchronization function within that scope, including the synchronization inside explicit reservation. It can wait for work that was already in flight. No synchronization is added for measurement.

`planning_us`, `buffer_us`, `binding_us`, `buffer_allocs` and `gpu_buffer_allocs` in recovery records are allocator-statistic deltas for the same scope. The real next batch's graph construction and binding occur afterward and are not recovery work; `binding_us` will normally be zero during reserve-only recovery. `buffers_before`/`buffers_after` use `[backend_id:bytes,...]`; aliased buffer types are counted once according to the existing allocator API. A physical backend buffer allocation during recovery also has `event=buffer scope=recovery`, with old and new chunk sizes. These sizes are not measurements of driver residency or total VRAM, and a GPU-owned host buffer type can be classified as GPU by the device-based counter.

For a comparison of allocation-path overhead, consider reallocation `path_us` together with recovery `recovery_us`, plus end-to-end server timings. Do not conclude that synchronization was eliminated just because it moved from `realloc_totals` to `recovery_summary`. These totals are cumulative, and the ordinary lifetime allocator statistics already include explicit recovery reservations; do not add those lifetime values a second time. Diagnostic bookkeeping and enabled graph fingerprints also perturb host timings.

After text/image/text, an expected bounded record has this form (illustrative values):

```text
sched_realloc_diag: event=recovery sched=6 context=2 role=main reason="restore_text_plan" recovery_count=1 success=1 recovery_us=900 sync_us=200 planning_us=300 buffer_us=0 binding_us=0 buffer_allocs=0 gpu_buffer_allocs=0 buffers_before=[0:428122116,1:407142600,2:0,3:289722400] buffers_after=[0:428122116,1:407142600,2:0,3:289722400] reserve_tokens=512 reserve_outputs=49 nodes=4512 leafs=992 splits=2 copies=1
```

The draft context should report its own scheduler/context ID and output bound. Necessary one-time backend/layout changes may remain. Repeated ordinary draft/catch-up size failures should stop once the canonical text plan is restored. Do not expect recovery counts to increase on every generated token.

## Collect a real Codex session

1. Start the instrumented server with level 1 and its unchanged production arguments. Capture startup through graceful shutdown. Keep the build commit, backend/device lines, model/quantization, context size, MTP/N-Gram parameters, batch/ubatch and pipeline configuration with the run.
2. Exercise an initial prompt, sustained generation, a follow-up using the prompt cache/checkpoint, and the small session-title requests that sometimes interrupt the main flow. Record rough boundaries and existing prompt/generation/speculative timing summaries. Do not change sampling or algorithms for this capture.
3. Extract the diagnostic and timing records from the log before sharing:

   ```bash
   rg 'sched_realloc_diag:|sched_graph_realloc:|print_timing|draft acceptance|drafted|graph reallocs|graphs reused' graph-realloc-level1.log > graph-realloc-level1-summary.log
   ```

4. Repeat a short representative portion with level 2 to inspect layout/topology alternation. If the first three samples miss a later trigger, use level 3 briefly. Use level 1 (and a matching disabled baseline) for throughput comparisons; level 2/3 add more observer work.

The most useful evidence is the `reason` and first failed tensor joined to `context/role`, `outputs/tokens`, previous/reserved graph characteristics, and `buffer_allocs`. Repeated `tensor_size` events with zero backend buffer allocations identify replanning rather than repeated GPU buffer creation. Alternating hashes suggest graph alternation; increasing required tensor bytes/positions suggest context-dependent storage. Neither observation alone establishes the final optimization.

## Validation

Recovery coverage adds two-buffer allocator/scheduler oscillation tests with one and multiple pipeline copies, plus small CPU Llama, Qwen35 and Qwen35-MTP context tests using F16 KV and Q8_0 KV with Flash Attention. They cover raw/mixed embeddings, the 49-output bound, checkpoint continuation in a separate context, graph reuse and retained scheduler/buffer size. The diagnostic-specific recovery tests cover levels 0/1/2/3, bounded output, failed recovery accounting and scheduler isolation without timing thresholds. These tests do not establish dual-GPU runtime correctness.

On the integrated branch, the CPU server and affected test targets built successfully. The five selected tests (`test-alloc`, `test-graph-recovery`, `test-batch-alloc`, `test-prompt-cache-ssd`, `test-responses`) passed with tracing disabled and at level 1; the allocator and recovery tests also passed at level 2. A level-3 run of the six model configurations reported twelve successful recoveries, unchanged compute-buffer sizes and zero additional backend buffer allocations during recovery. The allocator/scheduler tests passed under ASan/UBSan with LeakSanitizer disabled because of the execution environment. Before integration, the context regression was also checked against the unchanged production base and failed at the expected missing-recovery assertion. No Vulkan or ROCm runtime result is claimed.

The following measurements describe validation of the original diagnostic implementation, before recovery was merged:

`tests/test-alloc.cpp` contains model-free regressions for exact node-count, leaf-count, missing-allocation and tensor-size reasons, a same-count source-size reproducer, backend allocation failure accounting, aliased buffer-type accounting, and interleaved scheduler instances with tracing disabled and at levels 1/2/3. It checks backend assignment changes and isolation of final counters. The second scheduler uses pipeline copies.

Local validation on Linux x86-64 / GCC 13.3 / CMake 4.4:

- Release CPU builds of `llama-server`, `test-alloc`, `test-batch-alloc`, `test-prompt-cache-ssd`, `test-server-component`, `test-backend-ops` and `test-llama-archs` succeeded.
- The four model-free CTest regressions passed with tracing off and at level 2. The allocator test itself exercises levels 0/1/2/3, bounded detail counts, graph topology changes and automatic single-buffer replanning. GPU counter classification is exercised with the dummy backend, not a physical GPU.
- The generated tiny Qwen35 CPU architecture test passed with a fixed seed of 42, with tracing off and at level 2 (numerical comparison, save/load roundtrip and mixed batches).
- `test-backend-ops test -b CPU -o ADD,SUM -p 'type=f32'` passed 66/66 cases with level 2. No second hardware backend was available.
- The allocator test passed under AddressSanitizer and UndefinedBehaviorSanitizer with `ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1`. LeakSanitizer cannot run under this execution environment's ptrace setup, so no leak-check result is claimed.
- The repository code-style workflow's 158 model mappings and applicable EditorConfig encoding/newline/whitespace/space-indentation checks passed, as did `git diff --check`.
- Vulkan configuration was attempted but stopped at missing Vulkan headers/library/`glslc`. ROCm configuration was attempted but stopped at the missing HIP compiler/root directory. Neither GPU build nor GPU runtime is claimed.

A CPU scheduler-allocation microbenchmark compared the base and disabled diagnostic build, using identical compiler flags, two CPU backends, freshly reset tensor bindings, and no compute. Eleven alternating pairs of runs measured 1,000 allocations each after 50 warm-up calls. The graphs were synthetic ADD chains, not the production model's graphs:

| Synthetic nodes | Base median per allocation | Diagnostics disabled | Difference |
| --- | --- | --- | --- |
| 88 | 15.973 us | 15.883 us | -0.6% |
| 4512 | 739.036 us | 727.067 us | -1.6% |

These differences are within run-to-run variance; they do not establish a speedup. No disabled-path regression was detected in this check. The observer is passed as a constant null pointer on the normal allocator path so the compiler can remove diagnostic bookkeeping from the existing scan/binding code. End-to-end overhead on Vulkan/ROCm remains unmeasured.

Build the affected targets and run the model-free regressions:

```bash
cmake -S . -B build-diag-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_VULKAN=OFF -DGGML_HIP=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON
cmake --build build-diag-cpu --target llama-server test-alloc test-batch-alloc test-prompt-cache-ssd test-server-component -j "$(nproc)"
ctest --test-dir build-diag-cpu --output-on-failure -R '^(test-alloc|test-batch-alloc|test-prompt-cache-ssd|test-server-component)$'
```

Vulkan and ROCm use the same backend-neutral diagnostic path and require no backend-specific source changes. GPU compilation/runtime verification still requires the corresponding SDK, compiler and devices. Post-recovery overhead and behavior remain to be measured on the real workload; the supplied production log identifies the 88-node scheduler as MTP and the 4512-node scheduler as main through context labels, not by node count alone.
