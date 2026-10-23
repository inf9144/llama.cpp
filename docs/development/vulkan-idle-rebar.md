# Vulkan idle ReBAR experiment

Base: `codex-compat` at `f769a79b0404e1662cac5c259f454a2c4420b652`.
Branch: `feature/vulkan-idle-rebar-fastpath`.

This experiment adds an idle mapped-memory path to Vulkan `set_tensor_async` and `set_tensor_2d_async`, and removes the size limit from the existing pinned-host `cpy_tensor_async` optimization. Neither path has a size threshold, automatic cutoff, or size tuning option. The MoE cache and its SLRU policy are unchanged.

## Runtime switches

Options are read once per process. Use a fresh process for each configuration.

| Environment variable | Default | Effect |
| --- | --- | --- |
| `GGML_VK_IDLE_REBAR_FASTPATH` | `1` | Master switch; `0` disables both paths and their context tracking |
| `GGML_VK_IDLE_REBAR_SET` | `1` | Enable direct writes from `set_tensor_async` and its 2D variant |
| `GGML_VK_IDLE_REBAR_CPY` | `1` | Enable direct pinned-host copies from `cpy_tensor_async` |
| `GGML_VK_IDLE_REBAR_STATS` | `0` | Aggregate counters and host timings; print cumulative snapshots when a backend is freed |

All switches accept `0` or `1`. The master switch overrides the two path switches. Disabling the experiment uses the existing Vulkan transfer implementation, including its existing synchronization and staging behavior. It also disables the former small-copy CPU optimization: the old 128 KiB condition is completely gone. Therefore `off` is a transfer-only control, and the pinned base commit is the exact original behavior comparison.

## Safety and scope

The original `cpy_tensor_async` idle check only described the destination backend. It did not prove that another backend using the same Vulkan device had no recorded or submitted work. Equality of the transfer timeline's submitted/waited values also does not, by itself, prove completion.

Each Vulkan command context now takes a device-wide lease before recording its first command buffer. The lease survives submission, compute-context resets, event recording, and internal synchronization. It is released when the command context is destroyed. Backend contexts retain command contexts in their existing garbage collector until backend synchronization has completed and graph cleanup releases them. Temporary buffer-operation contexts remain alive until their synchronous fence wait returns.

The common direct-write helper requires all of the following:

- The allocation belongs to the destination Vulkan device, is `HOST_VISIBLE | HOST_COHERENT`, and has a non-null mapping.
- There are no live command-context leases on that device, across all backends and both compute and transfer queues.
- The caller has no compute context, transfer context, pending submission, or pending almost-ready fence, and has no outstanding transfer-timeline dependency.
- For `cpy_tensor_async`, the existing pinned-host admission check succeeds, and the source backend is synchronous or is the destination backend. Cross-device GPU copies keep their existing path.
- Offsets, source and destination pitches, and copy counts fit the mapped allocation and address arithmetic.
- `GGML_VK_SERIALIZE_SUBMISSIONS` is absent. Its existing synchronize implementation can return without waiting for a transfer-only submission, so the experiment disables direct writes and lease tracking in this mode rather than inferring completion from context lifetime.

The device mutex covers the idle check and the entire CPU copy. Taking a command-context lease uses the same mutex, so another backend cannot start recording work between the check and the copy. The `vk_buffer` shared pointer keeps the destination allocation alive during the call. The caller must keep source memory and tensors alive according to the existing async API contract and must supply dependencies between conflicting uses from different backends.

The helper calls the existing synchronous `ggml_vk_buffer_write_2d` mapped-memory implementation. It drains write-combined stores before releasing the mutex (`sfence` on x86, a sequentially consistent host fence elsewhere). The next Vulkan queue submission supplies the coherent host-to-device memory-domain operation; no flush of non-coherent memory or extra GPU wait is used. See the Vulkan specification's [host memory synchronization](https://docs.vulkan.org/spec/latest/chapters/synchronization.html#synchronization-submission-host-writes) and [memory-domain operations](https://docs.vulkan.org/spec/latest/appendices/memorymodel.html#memory-model-availability-visibility).

This is deliberately conservative: even completed GPU work can block the optimization until its command contexts are collected. An idle backend can use the fast path while other backend objects exist, provided none of them retains a command-context lease. This matters for main and draft model backends sharing a GPU. The experiment does not attempt range-level hazard tracking or poll fences to discover additional opportunities. The existing fallback is called without an added synchronize, queue wait, or device wait.

Mapped coherent memory can also be system memory or UMA memory. The counters identify the mapped-copy path, not proof of a physical PCIe BAR transaction. On the target AMD discrete GPUs the intended allocation is mapped device-local VRAM.

## Diagnostics

Enable `GGML_VK_IDLE_REBAR_STATS=1` for a separate diagnostic run. Statistics are per Vulkan device and split into `set` and `cpy`:

- `copies`, `bytes`, `cpu_us`: successful direct calls, payload bytes, and time in the CPU write plus store fence.
- `fallbacks`, `fallback_bytes`: host upload attempts rejected by the direct-write helper. `cpy` counts only sources admitted by the existing pinned-host check; device-to-device copies and sources rejected before that check are excluded.
- `fallback_host_us`: host time in the existing fallback. This includes recording and any staging or waits already performed by that path; it is not GPU DMA execution time.
- Rejections: `disabled`, `busy`, `not_host_visible`, `not_coherent`, `not_mapped`, `source_async`, `wrong_device`, `queue_mode`. Each rejected attempt has one reason, the first failed check. `queue_mode` identifies serialized submissions.
- `synchronize`: number of calls and host time inside `ggml_vk_synchronize`, including calls with no work. It does not cover every independent event or temporary-buffer fence wait.

Snapshots are cumulative for the process and device. Use the last snapshot for each device; do not sum snapshots from main and draft backends. There is no per-upload logging. Time counters have microsecond resolution and are only updated when statistics are enabled. Use statistics disabled for throughput measurements, then repeat with statistics enabled to explain the result.

## Build and correctness checks

Use the same compiler, Vulkan SDK, CMake options, GPU order, driver, and environment for both builds. From a clone containing this branch:

```bash
git worktree add --detach ../llama-rebar-baseline f769a79b0404e1662cac5c259f454a2c4420b652
cmake -S ../llama-rebar-baseline -B ../llama-rebar-baseline/build-vulkan \
    -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=ON
cmake --build ../llama-rebar-baseline/build-vulkan --target llama-server -j 8
cmake -S . -B build-vulkan -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build-vulkan --target llama-server test-backend-transfer test-backend-ops -j 8
ctest --test-dir build-vulkan -R test-backend-transfer --output-on-failure
build-vulkan/bin/test-backend-transfer Vulkan1
build-vulkan/bin/test-backend-ops test -b Vulkan0 -o CPY,CONT,ADD,MUL_MAT -p 'type=f32|type_a=f32'
```

`test-backend-transfer` checks pageable and pinned sources, view and call offsets, independent source/destination pitches, multi-row copies, payloads through 4 MiB, pinned CPU-source tensor copies, queued GPU reads before overwrites, 24 reuses of four cache-like slots, event dependencies, two backend contexts on one device, and concurrent transfers to disjoint tensors. GPU copies consume direct uploads before readback, so checking a host mapping alone cannot conceal missing device visibility. The five CTest cases cover both paths, neither path, set-only, cpy-only, and serialized-mode fallback. An unavailable requested backend returns skip code 77.

For Vulkan validation, use your SDK's validation layer and synchronization validation:

```bash
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT \
GGML_VK_IDLE_REBAR_STATS=1 build-vulkan/bin/test-backend-transfer Vulkan0
```

Repeat for `Vulkan1`. On hardware with non-host-visible VRAM, test the fallback and inspect its rejection counters. The existing `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` allocation preference can help, but does not guarantee non-host-visible allocations if the device exposes only mapped memory types. It is not a fast-path size option. Actual non-coherent/unmapped memory, two physical GPUs, and dedicated AMD transfer queues require hardware validation.

## Transfer microbenchmark

The optional microbenchmark reports CSV for pageable set, pinned set, and pinned tensor-copy calls at 4 KiB, 256 KiB, 1 MiB, and 4 MiB, both individually synchronized and in bursts of eight. Each case uses 32 repetitions. Wall time includes the existing synchronization needed to complete the transfers. It runs after the correctness checks and does not modify the MoE cache.

```bash
GGML_VK_IDLE_REBAR_FASTPATH=0 build-vulkan/bin/test-backend-transfer Vulkan0 --bench > transfers-off.csv
GGML_VK_IDLE_REBAR_FASTPATH=1 build-vulkan/bin/test-backend-transfer Vulkan0 --bench > transfers-on.csv
GGML_VK_IDLE_REBAR_CPY=0 build-vulkan/bin/test-backend-transfer Vulkan0 --bench > transfers-set.csv
GGML_VK_IDLE_REBAR_SET=0 build-vulkan/bin/test-backend-transfer Vulkan0 --bench > transfers-cpy.csv
```

Repeat on `Vulkan1`. For the exact base-commit microbenchmark, compile this new test source against the baseline ggml libraries, without changing baseline code:

```bash
baseline_bin_dir=$(realpath ../llama-rebar-baseline/build-vulkan/bin)
c++ -O2 -std=c++17 -Iggml/include tests/test-backend-transfer.cpp \
    -L"$baseline_bin_dir" -Wl,-rpath,"$baseline_bin_dir" -lggml -lggml-base -pthread \
    -o "$baseline_bin_dir/test-backend-transfer"
"$baseline_bin_dir/test-backend-transfer" Vulkan0 --bench > transfers-baseline.csv
```

These are upload-completion measurements, not model token throughput.

## Model A/B benchmark

`scripts/bench-vulkan-idle-rebar.py` starts and gracefully stops a fresh server for each case: `baseline`, `off`, `set`, `cpy`, `on`. It uses only the Python standard library. Supply the exact same production server arguments after `--`, including your current model and MTP files, tensor split, thread counts, batch sizes, KV types, and per-device MoE budgets. Omit host/port and authentication arguments. Example using local model paths and the existing production values:

```bash
# Bash; fill these with the exact values from the working production launcher.
MODEL=/absolute/path/to/model.gguf
DRAFT=/absolute/path/to/mtp.gguf
MOE_BUDGETS=YOUR_EXISTING_PER_DEVICE_MIB_LIST
TENSOR_SPLIT=YOUR_EXISTING_TENSOR_SPLIT

python3 scripts/bench-vulkan-idle-rebar.py \
    --baseline-bin ../llama-rebar-baseline/build-vulkan/bin/llama-server \
    --experiment-bin build-vulkan/bin/llama-server \
    --prompt-file /absolute/path/to/prompt-16k.txt \
    --prompt-file /absolute/path/to/prompt-64k.txt \
    --prompt-file /absolute/path/to/prompt-200k.txt \
    --repetitions 5 --warmup 1 --predict 512 --output-dir rebar-throughput \
    -- -m "$MODEL" --model-draft "$DRAFT" --spec-type draft-mtp \
    --device Vulkan0,Vulkan1 --tensor-split "$TENSOR_SPLIT" \
    --ctx-size 262144 --parallel 1 --load-mode mmap --lazy-mode on \
    --cpu-moe --moe-cache-mib "$MOE_BUDGETS" \
    --cache-ram 0 --cache-ssd-size 0 --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0
```

Keep any additional production arguments, including the MTP draft length, draft GPU placement, and ngram speculative modes. The example does not tune VRAM budgets or alter the existing cache policy. `llama-bench` currently does not expose the server's MoE-cache configuration, so the primary comparison uses `llama-server`.

Prompt files contain the exact raw completion prompts, including any desired chat-template tokens. Use representative prompts from real coding sessions and the identical files in every case. The script records prompt hashes, per-request timing and output hashes, logs, and median prefill/decode tokens/s. Native `/completion` requests disable prompt reuse, use seed 1234 and temperature zero, and ignore EOS to request exactly the specified number of decode tokens. Warmups are saved but excluded from medians. Truncated prompts or short generations fail the run. Check equal prompt and decode token counts and compare output hashes; GPU results can have numerical variation, so a hash mismatch needs investigation rather than automatically proving a transfer bug.

Repeat with `--stats` and a new output directory for upload/synchronization attribution. Repeat in reverse case order (`--cases on,cpy,set,off,baseline`) to check thermal and page-cache effects. Use identical power limits, GPU clocks, `RADV_PERFTEST`, device enumeration, mmap/lazy-loading, MTP configuration, and other background workloads. Inspect MoE hit/miss statistics and draft acceptance in the server logs. Treat the first model load/page faults separately from steady-state throughput; repeated prompts warm the expert cache but do not guarantee a particular hit rate.

## Local validation results

Validated on 2026-10-23 with Mesa 25.2.8 Lavapipe (LLVM 20.1.2), Vulkan 1.4.366 headers, and glslc 2023.8. This is software Vulkan with one queue, not the target AMD hardware.

- Built the Vulkan backend, full `llama-server`, and transfer/operator test executables.
- All five transfer CTest configurations passed, including source/destination views, strided copies, slot reuse, pending GPU reads, event waits, and two concurrent backend contexts. Counters confirmed direct set uploads and direct pinned tensor copies through 4 MiB, busy fallbacks for both callbacks, and zero direct copies with the master switch disabled.
- Vulkan synchronization validation reported no errors in the transfer suite.
- 351 selected `CPY`, `CONT`, `ADD`, and `MUL_MAT` cases passed against the CPU reference with the experiment on, with it off, and on the unmodified pinned base commit. The existing allocation test passed all 34 cases.
- All 24 microbenchmark combinations completed for baseline, off, and on. The same transfer test source was linked against the unmodified baseline libraries to verify the baseline command above. Software-memory throughput is not evidence of PCIe/ReBAR performance.
- The server benchmark runner passed a local mock smoke test covering five server lifecycles, 15 requests, warmup exclusion, result recording, and summaries. No real model inference benchmark was run here.
- Python compilation and `ty 0.0.88`, the repository model-naming style check, EditorConfig checker 3.0.3 on changed files, and `git diff --check` passed.

One initial parallel operator run printed `fatal library error, lookup self` after reporting success and exiting with status 0. Subsequent enabled, disabled, and baseline runs completed without that message. Its origin is unresolved; it was not a Vulkan validation error or a failed comparison.

## Remaining hardware questions

Large CPU writes hold the device mutex and occupy a CPU thread until the copy completes. They can reduce CPU/GPU overlap or perform worse than DMA, especially with cold mmap pages. No transfer-size cutoff hides that effect. Context tracking adds a mutex acquisition at context begin and release when enabled; compare `off` to the exact baseline as well as `on` to `off`. The device-wide lease can reject safe transfers to unrelated ranges, trading opportunities for a smaller synchronization proof.

Local software Vulkan tests cannot establish PCIe bandwidth, RDNA write-combining behavior, dedicated transfer-queue overlap, dual-GPU behavior, or model-level MTP/262144-context correctness and throughput. Those remain the purpose of the RX 7900 XTX / RX 7800 XT runs.
