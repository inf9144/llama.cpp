# MoE measurement baseline

Branch: `feature/moe-measurement-baseline`, based on merged `codex-compat` at `cbf40e9`.

This is an opt-in measurement build, not a new execution path or cache policy. No extra backend synchronization, GPU readback, transfer, kernel selection, admission, eviction, graph topology or allocator reservation is introduced. Existing readbacks are reused. Host records are buffered until graph submission finishes. Recording still costs CPU time and can affect overlap; use a separate uninstrumented reference process for throughput.

## Build

Keep your existing ROCm compiler, gfx1100/gfx1101, Flash Attention and all model/cache settings. Use separate build directories for reference and measurement so shared libraries cannot be mixed. Add only the following option to the measurement configuration:

```sh
cmake -S . -B build-rocm-moe-measure -DLLAMA_MOE_MEASURE=ON
cmake --build build-rocm-moe-measure -j 4
```

The first command is illustrative: pass the same HIP/toolchain/options as the existing build when creating a new directory. A fresh configuration does not inherit another directory's settings. `LLAMA_MOE_MEASURE` defaults OFF and is private to libllama; no runtime environment variable or CLI switch is added. OFF compiles all hooks out.

## Controlled collection

Do not change the fixed per-GPU cache budgets, model quantization, GPU split, batch settings, seed or prompt between reference and measurement. Record cold and warmed cache runs separately. Include actual target expert quantizations (IQ3_S may contain mixed tensor types), not just the Q8_0 draft.

Use actual microbatch widths 1, 4, 8, 32, 256, 512, 1024. A requested batch limit is not evidence that every microbatch has that width. The trace records the executed width, context and graph type. Obtain prefill-only and decode-only runs with the existing benchmark/application workload. Verify needs a genuine DFlash2 workload; ordinary four-token prefill is not verify. Capture both GPUs in the original production split first; separate single-GPU experiments are not directly equivalent to that workload.

The runner executes the command after `--` unchanged. For example, substitute your complete existing invocation:

```sh
python3 scripts/moe-measure/run.py --output results/moe-pp512-cold --phase prefill-only --note 'fixed cache budgets; cold; save full prompt/seed/split' -- build-rocm-moe-measure/bin/llama-bench YOUR_EXISTING_ARGUMENTS
python3 scripts/moe-measure/analyze.py results/moe-pp512-cold/trace.log --phase prefill-only --output results/moe-pp512-cold/host.json
```

`YOUR_EXISTING_ARGUMENTS` is a placeholder, not a runnable model configuration. Record the exact model and prompt identifiers. The runner refuses to overwrite directories, saves argv, revision, visibility variables, workload label, timestamps and exit status. Keep startup device enumeration to map ROCm ordinals to physical cards.

For a server workflow, use `mixed-dflash` and retain request/slot boundaries and acceptance logs. The script does not relabel each microbatch as decode or verify from its width. Warmup, prompt tail, mixed slots and draft contexts must be identified from the workload before phase-specific conclusions. It is preferable to start with isolated requests; mixed server traffic is a later experiment.

## Host trace coverage

`MOE_MEASURE` JSON lines include:

- `routing`: per-expert token-assignment histogram and host time of the **existing** ID readback/synchronization. Small direct-cache reads also capture pre-admission residency.
- `work_copy`: logical and backend allocation size, address and split identity of temporary weight destinations.
- `h2d`, `d2d`, `admission_h2d`: actual expert weight bytes queued by the existing callbacks, including the existing padding uploads. `enqueue_us` is host API/callback time, not GPU copy duration.
- `bank`: per-expert stride/type for every direct-cache weight bank, including all-hit batches.
- `reservation`: actual scheduler compute-buffer reservation per backend/context after graph submission.

Repeated weight banks share routing records by context/graph/layer. The analyzer rejects conflicting histograms and reports incomplete groups rather than inventing missing observations. Its per-split weight-address unions account for overlapping addresses. They are **not** a simultaneous cross-split live peak and cannot be subtracted from process VRAM to claim reclaimable cache space. Aliasing with activations and other scheduler copies can constrain actual savings.

Generic scheduler uploads not handled by these expert callbacks are outside host-counter coverage. GPU profiling captures those too. Host synchronization includes unfinished earlier work, not just the tiny ID copy. Its attribution must be checked against the device timeline.

## GPU time and memory

Use the installed ROCprofiler SDK. Its documented runtime trace collects HIP API, kernel, copy and allocation activity. A CSV collection supported by the parser is:

```sh
rocprofv3 --runtime-trace --output-format csv -- YOUR_EXISTING_COMMAND_AND_ARGUMENTS
python3 scripts/moe-measure/profile.py PATH_TO_ONE_COLLECTION --output gpu.json
```

Use a fresh directory for each collection and keep agent metadata, all raw files and the matching host trace. Current SDKs recommend native `rocpd` collection and conversion; if using that route, retain the native database and export the relevant CSV tables with the installed `rocpd convert` tool. Direct CSV is deprecated in recent SDKs, though the kernel/copy/allocation tables used here remain documented. See [AMD application tracing](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html).

The importer retains exact kernel names, agent identities, copy directions, summed durations and interval unions. MMQ/MMB/MMVQ names must be inspected against the actual dispatch; a quantized GEMM name alone does not establish MoE attribution. Kernel and transfer durations overlap, so their sums are not wall time. HIP graph replay complicates per-layer correlation; preserve correlation/graph IDs in the raw trace for analysis. No graph disabling or forced synchronization is added to simplify measurement.

Allocation/free events reconstruct process/agent allocation highwater by address; frees recover the allocation's original agent and size. This is HSA allocation accounting, **not** expert-only live bytes or total driver-reported VRAM. Unknown frees, missing tables, unsupported allocation operations or incomplete capture prevent a complete peak claim. A full-process profiler peak and an expert-copy address union answer different questions. Exact expert-only concurrent lifetime attribution and per-layer MMQ/MMB correlation require the real trace; these are not automatically solved by the host report.

## Offline capacity model

For each observed layer batch, the host report gives distinct experts, residency, assignments, copied/uploaded bytes and weight-bank sizes. Compact bank payload is `distinct_experts * sum(expert_bank_strides)`. For direct cache access, only observed misses need temporary slots. Reports give miss packet counts and capacities for 8, 16, 32 slots, including separately charged double buffering.

A slot here holds the **sum of observed Gate/Up/Down bank strides** for one expert. This is an explicit bundle model, not a claim that all banks must be simultaneously staged. Metadata, activations, alignment and allocator lifetimes are excluded from theoretical payloads. Keep cache contents and admission unchanged in this comparison. No simulated extra-cache hit-rate or speedup is claimed. All-hit workloads still reserve the configured staging capacity in a fixed-capacity architecture.

## Validation and first handoff

Run the small analysis tests:

```sh
python3 -m unittest discover -s scripts/moe-measure -p 'test_*.py'
```

Local validation uses CPU only. HIP/ROCm, profiler collection, production Swift/DFlash weights, PCIe transfers and real VRAM peaks are unavailable here. Before interpreting measurements, compare outputs/acceptance and uninstrumented/instrumented throughput on the same workload. Keep each complete collection, not only summaries: runner metadata, stdout, host trace, profiler data and the original reference timing. The first hardware collection should be one controlled request with both GPUs and fixed cache budgets; inspect coverage and phase attribution before running the whole matrix.

Completed local checks: CPU Release builds with measurement ON and OFF; the existing F16/Q8_0 QSA model/logit/checkpoint oracle passes both, with identical standard output. ON produces 96 graph/reservation events for that fixture and OFF produces none. That CPU fixture does not exercise real GPU expert transfers. A separate C++ writer smoke test validates JSON escaping, histograms and residency through the Python analyzer. Five Python tests cover byte models, alias interval unions, incomplete data, GPU interval overlap and allocation/free agent attribution. No GPU transfer timing or VRAM saving has been measured here.
