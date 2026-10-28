# QSA_SELECT candidate comparison

Starting point: the hardware-tested `e7b431d` on `feature/hip-qsa-sparse-attention`.

The user reports successful gfx1100/gfx1101 correctness, Vision, save/restore and rollback tests in the production model. At about 235K KV cells and decode width one, reported selection times are 144 us / 177 us, versus 50 us / 72 us for sparse attention. These are user measurements, not measurements from this build environment.

This commit adds test-only selection candidates and a comparison runner. **Inference still uses the exact e7b431d selection kernel and crossover.** No candidate is promoted before both GPUs pass exact comparisons and show a speed improvement. Attention arithmetic, indexer, Top-K, cache layout and state handling are unchanged. Crossover calibration is deferred until the selection winner is established.

## Candidates

The current budget is 512 pools of four cells plus three tail slots. Sorting 2051 slots rounds the old bitonic tile up to 4096; sorting the 2048 pool members separately avoids this doubling.

| Candidate | Work |
| --- | --- |
| `e7b431d-bitonic` | Frozen original kernel: padded bitonic network and the original block compaction. Also checked against the actual GGML backend operator. |
| `members-bitonic-tail` | Gather each pool's score once, bitonic-sort just the members, merge the three actual tail cells, use Wave32 prefix compaction. |
| `pool-merge-tail` | Sort each pool's four actual cell addresses in registers; merge the sorted runs by binary ranks; merge the tail and compact. |
| `hipcub-radix-tail` | Optional SDK block radix sort over the pool members, restricted to the significant bits of `n_kv`; merge the tail and compact. No device-wide sort, allocator or persistent scratch. |

Every candidate sorts **cell values**, not pool IDs. No contiguous or monotone physical KV layout is assumed. A permutation of the entire KV address space is explicitly tested. Pool runs are merged with lower/upper bounds that disambiguate duplicate ranks. Tail duplicates and overlap with selected pools are removed during the final compaction. Invalid cells and invisible pools have the same semantics as the original. The output is exactly the original sorted unique I32 list with `-1` padding.

The compaction computes per-thread counts over contiguous sorted spans, scans within logical Wave32 groups, then scans eight wave totals. It removes the original repeated block-wide scan barriers. The comparison covers both the tail-padding optimization and a change of sorting algorithm; it does not assume either is faster before timing.

hipCUB is used only if its SDK header is present; otherwise the radix candidate is reported as unavailable. It is not a new required dependency. Its block radix API is documented in the [official hipCUB reference](https://rocm.docs.amd.com/projects/hipCUB/en/docs-5.6.0/.doxygen/docBin/html/classhipcub_1_1BlockRadixSort.html).

## Local validation

`test-qsa-select` runs 160 host tests comparing the shared pool/tail merge ranks and local four-element sorting against `std::sort`, including repeated values and all-invalid padding. These tests are independent of GPU timing.

The GPU suite first runs 28 correctness cases with 1-1023 picked pools, widths 1/3/8, nonzero view offsets, strided rows, arbitrary cell permutations, overlapping/partial pools, duplicate picks/members/tails, invalid indices, invisible/NaN scores and empty selections. Every candidate is checked byte-for-byte against both the actual backend `QSA_SELECT` and an independent host `std::sort`/unique union. Guard regions detect writes outside the output. Captured candidates are checked again after graph execution.

The comparison then uses 235520 KV cells, 512 pools, widths 1-8/32/512/1024 and both linear and permuted cell layouts. This is only the point-1 selection experiment; it does not add the point-2 context-boundary matrix yet.

## Run on each GPU

Using the existing HIP build configuration:

```sh
cmake --build build-rocm10 --target test-qsa-select
build-rocm10/bin/test-qsa-select --backend ROCm0 > qsa-select-ROCm0.log 2>&1
build-rocm10/bin/test-qsa-select --backend ROCm1 > qsa-select-ROCm1.log 2>&1
```

Run the two commands sequentially, without concurrent inference, and retain the complete logs. The program prints the actual device, gfx architecture, HIP runtime and available candidates; ROCm ordinal alone is not treated as evidence of a particular GPU.

Each timing uses a graph containing 32 identical selection launches to avoid CPU enqueue gaps dominating a fast candidate. GPU event time and amortized graph wall time are reported separately. There are 21 samples per candidate; candidate order rotates between rounds. `gpu_median_us`, `gpu_p95_us`, `graph_wall_us` and speedup against the frozen baseline are printed. The standalone actual GGML operator's wall time is also reported as `baseline_ggml_wall_us`; it includes framework overhead and must not be confused with the event timing.

All numerical checks occur outside timed regions. The graph outputs are checked before collecting timing samples. These are selector measurements, **not** attention, complete QSA-subgraph or end-to-end inference timings. Point 2 must use the actual promoted kernel and measure the complete QSA subgraph at the requested context boundaries.

## Promotion criteria and remaining work

- Both physical GPUs must complete every exact-list and guard check, including captured execution.
- Compare median and p95 against the original on both linear/permuted layouts and decode/verify/prefill widths. Do not promote a candidate on the strength of decode width one alone.
- Select one simple variant with a consistent demonstrated gain; keep the old implementation for any unsupported shape if needed.
- After promotion, rerun the existing GPU sparse-attention/Decode-Verify numerical checks and the production-model checks.
- Only then extend the full QSA-subgraph benchmark to 8K/12K/16K/24K/32K/48K and calibrate the automatic crossover.

This environment has neither ROCm SDK nor `/dev/kfd`. Host tests and the existing CPU QSA oracles can run here; the new HIP candidates cannot be compiled, validated or timed here. There are deliberately no invented speedup numbers or changed production thresholds.
