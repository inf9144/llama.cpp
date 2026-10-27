# RDNA3 kernel port

Base: `codex-compat`, `5239d46903a1dcbf3a6ca6accd9f35be63ab04fa`.
Target: ROCm/HIP, RX 7900 XTX (`gfx1100`) and RX 7800 XT (`gfx1101`).
This is a source port for review; GPU correctness and performance still require the target cards.

## Sources and scope

| Reference | Ported | Existing implementation retained / omissions |
| --- | --- | --- |
| [0003](https://github.com/stew675/llama.cpp/commit/d09fc6a35685414de8a22ca8e46462f4d86446b7) | Chunked scalar-gate GDN prefill, gfx11 BF16 WMMA for head 128, FP32 fallback; pre-batch SSM and convolution snapshots for complete-batch rollback. | Existing sequential/KDA and upstream GDN MMA kernels remain. The upstream HIP MMA selector targets RDNA3.5, so it does not cover these discrete gfx11 cards. Multi-sequence batches use the existing implementation. No new recurrent-cache format or snapshot-validity tracker. |
| [0005](https://github.com/stew675/llama.cpp/commit/0484909cc4220793eee6c9430e6fb93d33097560) | Native Q8_0 K/V tile loads; common allocation/launch policy; gfx11 decode/verify geometry and split heuristic independent of widths 1..8; derived KQ mask and fully masked prefill-group skipping; masked-value zeroing in WMMA. Existing Q4/Q5/BF16 KV layouts retain the stock conversion/vector fallback; their additional native loaders are omitted. | Existing vector kernels, supported KV types and head shapes retained. Q8 fetch-ahead and the special decode WMMA band are RDNA4-specific in the reference and are omitted. No IQ4_NL KV extension, head-512 tuning or RDNA2/APU tuning. |
| [0006](https://github.com/stew675/llama.cpp/commit/0d3043303a46be5a04786c02db363c18ef8c7e38) | BF16-activation MMB quantized prefill GEMM for dense and routed experts, including fused normal SwiGLU; device-explicit eligibility and existing-pool scratch. IQ3_S and Q8_0 are covered. | Existing MMQ/rocBLAS/MMVF paths handle unsupported shapes, small batches, noncontiguous inputs and other architectures. MMF32 is omitted: no measured gfx11 benefit is available. No producer conversion cache, expert cache, shadow weights or allocator hooks. |
| [0007](https://github.com/stew675/llama.cpp/commit/16b142a72a3d33c165224bc71b7d22a7ad2416c2) | MMVQ expert VDR for Q8_0/Q4_K/Q5_K, Q6_K dot products, IQ3_S sign unpacking, small integer multiplies, K-quant scale reuse, batched MMVF reductions and gfx11 grid-stall padding with a dummy-block bounds guard. | Existing dense VDR policy and upstream MMVQ expert launcher retained. No MMQ tuning environment variables, RDNA2 tuning, RDNA4 short-row kernel or unrelated Q1_0 changes. |

### Dispatch and rollback details

* GDN processes an aligned prefix, then at least 16 tokens sequentially. The tail retains every final `K` snapshot, including fused-cache destinations. For `K <= 16`, the prefix boundary is identical with and without rollback slots.
* `n_rs_batch` carries the existing speculator's maximum verify width into the graph; it adds no CLI flag or environment variable. Verify batches up to this bound stay sequential. API callers leaving it zero use snapshot depth as the bound. Rebuild native API clients against the changed context-parameter struct and GDN operation signature.
* MMB selects the actual `ctx.device`, requires contiguous operands, `K % 256 == 0`, `M % 128 == 0` and at least 512 activation rows/tokens. Routed descriptors stay within the stock IDs helper limits. Negative expert IDs leave zero output. Conversions and descriptors live only in the existing operation scratch pool.
* Derived masks are probed automatically on a CUDA/ROCm device. Only single-stream, single-sequence prefill without ALiBi or nondegenerate M-RoPE is eligible. Meta devices, unsupported backends and other shapes keep packed masks. The graph reserves packed-mask fallback space where reachable. Vulkan and the other FA backends without compact-mask loaders explicitly reject derived-mask nodes.
* All GDN rollback state, stock MoE caching, SSD prompt-cache, DFlash2 vision and tail-ingest interfaces are retained. No Stew MoE cache, slab allocator, dynamic VRAM budget or persistent conversion cache is added.
* FA template instances are split by head/type or head/geometry. The generator is byte-reproducible (238 files) and emits matching extern declarations to avoid duplicate implicit kernel instantiations and bound compiler memory. WMMA device bodies for heads above 256 are discarded because the existing RDNA selector never uses them; the host launcher asserts that limit.

## ROCm host staging pool

The pool is enabled by default on ROCm/HIP with a 512 MiB budget per backend context and no tensor filter. No environment settings are required. It copies async host-to-device uploads into private pinned host buffers before submitting them to HIP. It reuses completed buffers, waits for the oldest submission under pressure, and splits uploads larger than the budget into chunks of at most 64 MiB. If a pinned allocation fails, subsequent allocations use pageable malloc buffers; existing pinned buffers remain usable. Completion events protect every buffer until its transfer finishes.

Optional overrides:

| Variable | Default | Effect |
| --- | --- | --- |
| `GGML_ROCM_MALLOC_STAGE_POOL` | `1` | Set to `0` to disable staging. |
| `GGML_ROCM_MALLOC_STAGE_POOL_MIB` | `512` | Host buffer budget per backend context, clamped to 1..4096 MiB. |
| `GGML_ROCM_MALLOC_STAGE_POOL_TENSOR` | unset | Restrict staging to tensor names containing this substring. |

For example, `set -gx GGML_ROCM_MALLOC_STAGE_POOL_TENSOR ffn_` restricts staging to FFN tensor names in fish. Remove that restriction with `set -eg GGML_ROCM_MALLOC_STAGE_POOL_TENSOR`. Synchronous uploads, 2-D uploads, device-to-device copies and other backends keep their existing paths. Configuration is read once per process.

Only these three pool settings are supported. The production integration excludes the checkpoint diagnostics, GALLOCR identity/signature checks and invalidation workaround, upload controls, reproducer tool, trace switch and periodic pool statistics. Normal graph allocation recovery needed by embedding and vision paths is retained.

## Validation in this environment

The container has no `/dev/kfd` or `/dev/dri`, no AMD GPU and no production model weights.

| Check | Result |
| --- | --- |
| CPU Release build, relevant test targets | Passed; rebuilt after frontend changes. |
| DFlash vision, tail-ingest and disabled-tail CTests | 3/3 passed. |
| Graph allocation recovery CTest | Passed. |
| Derived-mask host oracle | Passed: 18 exact packed-fill comparisons covering SWA variants, causal/non-causal modes, holes and foreign cells; ineligible batch guards also checked. |
| Derived-mask FA backend reference cases | 13/13 CPU cases passed, including ragged KV and interior holes. |
| Quant sign transform algebra | 69,632 actual grid-table/sign combinations matched independent byte-wise signed negation exactly. |
| GDN backend reference cases | 15/15 focused CPU cases passed (10 normal, 5 fused-cache destinations), including rollback slots and long prefill shapes. This does not execute the new HIP kernels. |
| Quantized dense / routed / fused-SwiGLU backend cases | 30/30 CPU reference cases passed; target GPU runs remain pending. |
| Qwen3.5 recurrent context tests | 7/7 passed: 6 existing cases plus the new full-verify rollback test at widths 1..8 (exact logits). The new test reproduces a failure at width 1 against the unchanged base library. |
| Strict logits-width probe, CPU, Q8_0 KV, prefix 128 | Width 1 differs from width 8 by at most `7.45058e-8`; widths 2..8 are bit-identical. Strict probe exits 2. The unchanged base commit gives exactly the same result. This is a known baseline result, not a passing bit-identity test. |
| Vulkan Release build | Passed, including generated shaders and `test-backend-ops`. No Vulkan device was enumerated, including the attempted software ICD; runtime GPU checks were skipped. Other backend SDKs were unavailable for builds of their compact-mask rejection guards. |
| HIP build for `gfx1100;gfx1101` | Passed: backend, llama-server, llama-bench, test-backend-ops, test-llama-context, test-llama-archs and the width probe compiled and linked. The complete build used Release host flags and `-O1` for HIP device compilation to bound container resources. Ten relevant translation units additionally passed `-O3` compilation for both targets (GDN FP32/BF16, MMB, MMVF/MMVQ, IQ3_S MMQ, Q8_0 FA tile heads 128/256 and two head-256 MMA geometries). |
| CPU llama-bench smoke comparison | Four alternating base/port runs passed with the synthetic Qwen3.5 model, Q8_0 KV, pp2048/tg256, one thread and five repetitions. Concurrent HIP compilation caused substantial variance; these runs do not establish performance equivalence. |
| HIP-linked host smoke tests | 5/5 CTests passed (DFlash vision/tail/disabled-tail, graph recovery, derived-mask oracle); the full-rollback case also passed using CPU fallback. The server version command passed. ROCm device enumeration found no GPU; the backend test selected zero GPU cases and is recorded as skipped. |
| GPU numerical tests / DFlash acceptance / model perplexity | Not executable here. |
| GPU benchmarks, 262144 context, mixed-card multi-GPU | Not executable here; no speedup or absence of GPU regressions is claimed. |

Only HIP GDN head-128 long-prefill uses `1e-4` NMSE for BF16 accumulation; other backends keep their original tolerance, and short decode/verify retains the existing tighter tolerance. Backend matrix tests use the existing quantized GEMM tolerances. These tolerances are test criteria, not measurements on the target hardware.

## Reproduce on the cards before merge

Build with the normal installed ROCm SDK (the container used ROCm 7.2.4 and Clang 22):

```sh
cmake -S . -B build-hip -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release \
  '-DCMAKE_HIP_ARCHITECTURES=gfx1100;gfx1101' -DLLAMA_BUILD_TESTS=ON
cmake --build build-hip -j
ctest --test-dir build-hip -R '^test-(dflash|kq-derived)' --output-on-failure
build-hip/bin/test-backend-ops -b ROCm0 -o GATED_DELTA_NET,GATED_DELTA_NET_CACHE_FUSION
build-hip/bin/test-backend-ops -b ROCm0 -o FLASH_ATTN_EXT
build-hip/bin/test-backend-ops -b ROCm0 -o MUL_MAT,MUL_MAT_ID,MUL_MAT_ID_GLU
build-hip/bin/test-backend-ops perf -b ROCm0 -o GATED_DELTA_NET,FLASH_ATTN_EXT,MUL_MAT,MUL_MAT_ID
build-hip/bin/test-logits-width-probe model.gguf prompt.txt 4096 1024 q8_0
```

Use the enumerated backend names (`ROCm0`, `ROCm1`) and check that cases actually run: the test harness reports unmatched backends as skipped. Run each card separately and both together. The decode/verify geometry guard covers widths 1..8; wider batches keep existing family selection and need separate validation at the actual draft width. The width probe uses the same tokens and query rows in independently initialized contexts of widths 1..8; it exits 2 for any bit difference and reports maximum absolute error. It also varies recurrent snapshot depth while keeping the verify-width bound fixed.

Compare the same commands and production `llama-bench` workloads against the pinned base, including IQ3_S target, Q8_0 draft/KV, long prefill and 262144 context. Include DFlash2 generation with vision and tail-ingest, partial and complete verify rollback, and SSD prompt-state restore. Reject reproducible throughput or numerical regressions before merging.
