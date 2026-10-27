# RDNA3 fusion and DFlash2 device-input port

Base: `codex-compat`, `1d8511dbd93ef5157a41224ef7a329f46a4eb232`.
Review branch: `feature/rdna3-fusions-qsa-dflash`.
Targets: ROCm/HIP, RX 7900 XTX (`gfx1100`) and RX 7800 XT (`gfx1101`).

This is a selective port for hardware review. CPU reference tests pass; the new HIP kernels, GPU transfer ordering, DFlash acceptance and performance are not validated on the cards in this environment.

## Scope by reference

| Reference | New changes | Existing coverage and omissions |
| --- | --- | --- |
| [0008 - fused kernels](https://github.com/stew675/llama.cpp/commit/953795c14c71815c53fad8d3e4d2589309daba1d) | Narrow-row RMS normalization, including vector weight and scalar scale forms; RMS + weight + sigmoid-gate fusion for GDN. | Dense/routed Gate+Up+GLU, router fusions and MMQ fusion already exist in the base, including the earlier MMB port. No duplicate dispatcher. Norm/Q8_1 producers that rely on a context-level conversion cache are omitted. Additional SwiGLU-to-quantizer variants are deferred pending HIP quantization/acceptance checks; existing per-operation quantization is retained. |
| [0010 - dispatcher/infrastructure](https://github.com/stew675/llama.cpp/commit/0b0e3735ac2904db8f253388c1b3ecdfb6267361) | Slab-independent target-to-draft device feature transfer, internal graph-mode separation and reader completion events. | Existing RMS/weight/RoPE/scatter, RMS/scale, convolution/bias/SiLU and recurrent snapshot-copy fusions remain. No slab hooks, input rings, scheduler replacement, forced graph invalidation or extra diagnostics. |
| [0011 - Qwen/HC/QSA](https://github.com/stew675/llama.cpp/commit/3e74e289dc399ab30537864f9b0e94d3d88e2c52) | Order the independent ROCm GDN gate projection before normalization so the sigmoid fusion can match. | HC-averaged DFlash layer inputs, HC pre/post/combination and the four-head FP32 lightning indexer are already implemented upstream. Retain these implementations. Do not add parallel HC/QSA operators. QSA3 and the four-head WMMA indexer are omitted on gfx1100/1101; see below. |
| [0012 - DFlash2/other](https://github.com/stew675/llama.cpp/commit/33eeb031d3576b6ed1009bfaf3633226ab20901a) | Automatic DFlash2 device input selection with host fallback, row permutation handling, embedding-equivalent KV metadata and on-demand host feature access. | Prompt-tail ingest and vision gap filling already exist and are retained. Current GPU TopK already uses radix/bitonic selection. No replacement by an older CPU TopK variant, Meta all-gather subsystem, allocator, parking, lazy-reader or MoE-cache changes. |

Stew's `fattn-qsa3.cu` eligibility policy reports a roughly 13% slowdown against its vector path on a dual RX 7900 XTX setup at d64K, and defaults QSA3 off on RDNA3.0. Its four-head WMMA indexer is also opt-in on gfx1100. The measured/default-enabled paths target gfx1151/RDNA4. Neither is enabled here without gfx1100/1101 measurements. No opt-in switches are imported.

The direct CONCAT/SSM_CONV kernel is also omitted. The reference documents unresolved numerical divergence under multi-device tensor splitting; importing its global single-GPU gate would disable it on this two-card machine, while removing the gate would require GPU validation. Existing convolution/SiLU and recurrent-copy fusions remain in use.

## Automatic dispatch and lifetimes

* Narrow norms require an RDNA3.0 wave32 device, F32 rows with 32..256 columns in multiples of 32, contiguous/aligned columns and aligned row/channel/sample strides. A vector weight must broadcast over all rows. Plain RMS, weighted RMS and RMS/scale use the grouped-row path from 4096 rows; the four-operation sigmoid fusion also covers short decode/verify batches. Unsupported layouts and observable intermediates retain existing kernels.
* Eight rows share a 256-thread block, with a bounds guard for the final partial group. The adaptation preserves the base's **float4** partial sums and XOR reduction tree, rather than copying Stew's older scalar reduction. Gate multiplication uses the same sigmoid expression as the existing unary kernel. Bit identity on HIP remains to be checked.
* DFlash2 device inputs are selected for one sequence, a physical ROCm/CUDA device holding the draft's feature projection, and a compatible target backend on that device. Layer splitting across multiple GPUs is allowed. Tensor/Meta splitting, other backends, unsupported configurations and failed staging allocation retain host injection. Vulkan keeps its existing host input path and graph ordering.
* The target holds F32 feature tensors of shape `[n_embd, n_batch]` on the projection device. Additional device memory is approximately `4 * selected_layers * n_embd * n_batch` bytes plus alignment. These buffers contain only the current target batch. There is no prompt, weight or conversion cache and no dynamic memory budget.
* Each layer copies from its actual source backend to the staging backend. The draft waits for the target copies; completion events cover every draft backend that can read or copy the features. The next target batch waits for those events before overwriting the buffer. Backends without events synchronize their reader instead.
* The original-to-extracted row map survives logits access and recurrent microbatch reordering. Host layer getters download and reorder a staged layer only when requested. If extraction becomes ineligible, earlier captured microbatches are materialized before falling back.
* Ingest row indices are an internal graph mode, never vocabulary tokens in KV metadata. The graph reuse predicate distinguishes them from a same-width noise token batch. Standard embedding/vision graph allocation recovery remains; no second scheduler or graph-rebuild workaround is added.
* Vision gaps still use zero-feature embedding injection. Prompt-tail trimming remains before target feature access. The existing stock MoE cache, SSD state format and ROCm malloc staging pool defaults are unchanged. No CLI options or environment variables are added.

## Validation

| Check | Result |
| --- | --- |
| CPU Release build: library, common code, test-llama-archs, test-backend-ops | Passed (GCC 13.3, CMake 4.4). |
| DFlash2 host/device input oracle on CPU | Passed with F16 and Q8_0 KV: permuted/interleaved sequences, multiple target microbatches, 7/1/4-token transitions, exact raw features, byte-identical serialized KV including metadata, identical selector lattices, invalid row rejection and mode reset. Automatic CPU activation stays disabled. |
| Existing HC layer extraction, graph recovery, DFlash vision, tail and disabled-tail CTests | Passed, together with the new oracle: 6/6. |
| Norm-gate backend reference cases on CPU | 20/20 passed: widths 32/64/96/128/160/256, row counts 1/8/4099, strided input and an externally consumed intermediate. These do not execute the HIP fusion. |
| Narrow RMS reference cases on CPU | 7/7 passed, including separate channel/sample dimensions and a partial final row group. These do not execute the HIP grouped-row kernel. |
| Source/whitespace review | `git diff --check` passed. |
| HIP build for gfx1100/gfx1101 | Not executable: CMake cannot find a HIP root/compiler; no ROCm SDK is installed in this environment. This is separate from the earlier kernel-port build recorded in rdna3-kernel-optimizations.md. |
| Vulkan build/runtime | Not executed here; SDK/device unavailable. No Vulkan backend code changes or automatic device-input activation. |
| GPU numerical tests, asynchronous event ordering, acceptance and benchmarks | Open: no GPU, production model weights or 262144-context run available. No GPU speedup or absence of regressions is claimed. |

## Targeted checks on the cards

Build against the local ROCm SDK for both cards using the existing build settings. Then run the relevant operator comparisons on each device:

```sh
build-rocm10/bin/test-backend-ops -b ROCm0 -o RMS_NORM_GATED
build-rocm10/bin/test-backend-ops -b ROCm1 -o RMS_NORM_GATED
build-rocm10/bin/test-backend-ops -b ROCm0 -o RMS_NORM -p '4099|683'
build-rocm10/bin/test-backend-ops -b ROCm1 -o RMS_NORM -p '4099|683'
ctest --test-dir build-rocm10 -R '^test-(dflash-device-inputs|qwen4exp-layer-inputs|dflash-vision|dflash-tail|dflash-tail-disabled|graph-recovery)$' --output-on-failure
```

The model CTest oracle deliberately uses CPU-backed tensors; it checks transfer representation and graph semantics, not HIP execution. Hardware validation must additionally compare the base and review branch with the actual IQ3_S target, Q8_0 DFlash2 and Q8_0 KV: identical prompt/seeds, decode versus speculative verify, vision/tail transitions, draft acceptance and short/long-context throughput. For the sigmoid fusion, compare the same GPU cases with the pre-existing `GGML_CUDA_DISABLE_FUSION=1` override as well; no new production switches are provided. The standalone grouped-row norm is selected independently of graph fusion and should be compared against the base branch.
