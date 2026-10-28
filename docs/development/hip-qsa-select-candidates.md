# QSA_SELECT candidate comparison

Starting point: the hardware-tested `e7b431d` on `feature/hip-qsa-sparse-attention`.

The user reports successful gfx1100/gfx1101 correctness, Vision, save/restore and rollback tests in the production model. At about 235K KV cells and decode width one, reported selection times are 144 us / 177 us, versus 50 us / 72 us for sparse attention. These are user measurements, not measurements from this build environment.

The comparison runner was published at `4303b95` with inference still using e7b431d. The two GPU logs supplied on 2026-10-28 now establish `hipcub-radix-tail` as the winner, so the measured configuration uses it automatically. Attention arithmetic, indexer, Top-K, cache layout, state handling and the crossover are unchanged. The boundary matrix below is the next measurement step.

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

hipCUB is used only if its SDK and rocPRIM headers are present; otherwise the production kernel remains bitonic and the radix candidate is unavailable. It is not a new required dependency. Its block radix API is documented in the [official hipCUB reference](https://rocm.docs.amd.com/projects/hipCUB/en/docs-5.6.0/.doxygen/docBin/html/classhipcub_1_1BlockRadixSort.html).

## Local validation

`test-qsa-select` runs 160 host tests comparing the shared pool/tail merge ranks and local four-element sorting against `std::sort`, including repeated values and all-invalid padding. These tests are independent of GPU timing.

The GPU suite first runs 28 correctness cases with 1-1023 picked pools, widths 1/3/8, nonzero view offsets, strided rows, arbitrary cell permutations, overlapping/partial pools, duplicate picks/members/tails, invalid indices, invisible/NaN scores and empty selections. Every candidate is checked byte-for-byte against both the actual backend `QSA_SELECT` and an independent host `std::sort`/unique union. Guard regions detect writes outside the output. Captured candidates are checked again after graph execution.

The comparison then uses 235520 KV cells, 512 pools, widths 1-8/32/512/1024 and both linear and permuted cell layouts. This command remains the point-1 selection comparison. The separate `--crossover` command below measures the requested context boundaries.

## Run on each GPU

Using the existing HIP build configuration:

```sh
cmake --build build-rocm10 --target test-qsa-select
build-rocm10/bin/test-qsa-select --backend ROCm0 > qsa-select-ROCm0.log 2>&1
build-rocm10/bin/test-qsa-select --backend ROCm1 > qsa-select-ROCm1.log 2>&1
```

Run the two commands sequentially, without concurrent inference, and retain the complete logs. The program prints the actual device, gfx architecture, HIP runtime and available candidates; ROCm ordinal alone is not treated as evidence of a particular GPU.

Each timing uses a graph containing 32 identical selection launches to avoid CPU enqueue gaps dominating a fast candidate. GPU event time and amortized graph wall time are reported separately. There are 21 samples per candidate; candidate order rotates between rounds. `gpu_median_us`, `gpu_p95_us`, `graph_wall_us` and speedup against the frozen baseline are printed. The standalone actual GGML operator's wall time is also reported as `production_ggml_wall_us` (`baseline_ggml_wall_us` in the original uploaded logs); it includes framework overhead and must not be confused with the event timing.

All numerical checks occur outside timed regions. The graph outputs are checked before collecting timing samples. These are selector measurements, **not** attention, complete QSA-subgraph or end-to-end inference timings. Point 2 must use the actual promoted kernel and measure the complete QSA subgraph at the requested context boundaries.

## Measured winner and production dispatch

Source: the complete user logs `qsa-select-ROCm0(2).log` and `qsa-select-ROCm1(2).log`, produced by the `4303b95` comparison runner. Both identify the physical GPU, report hipCUB available and finish successfully. Each passes 28 byte-exact correctness/guard cases and 22 timed configurations, including linear and fully permuted cell layouts. Radix beats every other candidate in both median and p95 in all 22 configurations on each GPU.

Representative GPU medians at 235520 KV cells and 512 picked pools, linear layout:

| GPU | Query width | e7b431d bitonic (us) | Radix (us) | Selection speedup |
| --- | ---: | ---: | ---: | ---: |
| gfx1100 / RX 7900 XTX | 1 | 114.227 | 9.475 | 12.056x |
| gfx1100 / RX 7900 XTX | 8 | 114.071 | 9.549 | 11.946x |
| gfx1100 / RX 7900 XTX | 512 | 290.399 | 53.010 | 5.478x |
| gfx1100 / RX 7900 XTX | 1024 | 525.565 | 90.116 | 5.832x |
| gfx1101 / RX 7800 XT | 1 | 177.266 | 11.604 | 15.276x |
| gfx1101 / RX 7800 XT | 8 | 175.956 | 11.639 | 15.118x |
| gfx1101 / RX 7800 XT | 512 | 443.216 | 80.665 | 5.495x |
| gfx1101 / RX 7800 XT | 1024 | 849.813 | 167.777 | 5.065x |

Across all measured layouts and widths, selection speedups range from 4.940x to 12.067x on gfx1100 and from 4.646x to 15.335x on gfx1101. These are isolated selection timings, not model throughput or full QSA speedups.

`ggml/src/ggml-hip/qsa-select.cuh` now holds the measured radix, tail merge and compaction implementation, shared by the backend and candidate runner. Production dispatch selects its 2048-member instantiation only on gfx1100/gfx1101, with four cells per pool, 512 picked pools and three tail slots. Other pool budgets/shapes/devices and builds without the headers retain the original bitonic implementation. The original attention kernels and model crossover are source-identical to e7b431d. No host synchronization, persistent scratch allocation, inference flag or cache is added. Sorting still uses actual cell addresses; neither pool order nor KV contiguity is assumed.

Local validation after promotion: the CPU Release targets build, 160 host merge tests and all three targeted CTests (`test-qsa-select`, `test-qsa-sparse`, `test-qwen4exp-qsa-equivalence`) pass. Source comparisons verify that radix/compaction match the measured candidate and that fallback/attention code is unchanged. A small CPU smoke run also exercises the new component/complete-graph timing harness with ranked selections, causal tails and permuted cells; its times are not GPU performance results.

This environment still has neither ROCm SDK nor `/dev/kfd`. The promoted backend integration has not been compiled or executed here on HIP. The supplied candidate logs validate the radix implementation on both cards. The subsequent `qsa-numeric-ROCm0.log` and `qsa-numeric-ROCm1.log` also validate the promoted backend integration: ten cases per GPU pass, including nine exact decode/verify cases at widths 1-8 and a strided two-stream case. Maximum NMSE is 7.91824e-6 on gfx1100 and 8.17522e-6 on gfx1101, below the 5e-4 bound. Production-model DFlash acceptance after promotion remains a local check, even though the index list is byte-identical and attention arithmetic is unchanged.

## Crossover measurement after promotion

Use the existing HIP build configuration and run sequentially without concurrent inference:

```sh
cmake --build build-rocm10 --target test-qsa-sparse test-qsa-select
build-rocm10/bin/test-qsa-sparse --backend ROCm0 > qsa-numeric-ROCm0.log 2>&1
build-rocm10/bin/test-qsa-sparse --backend ROCm1 > qsa-numeric-ROCm1.log 2>&1
build-rocm10/bin/test-qsa-sparse --backend ROCm0 --crossover > qsa-crossover-ROCm0.log 2>&1
build-rocm10/bin/test-qsa-sparse --backend ROCm1 --crossover > qsa-crossover-ROCm1.log 2>&1
```

`--crossover` is an option of the test binary only. It does not change inference dispatch. It covers 8192, 12288, 16384, 24576, 32768 and 49152 KV cells; widths 1-8, 32, 512 and 1024; linear and fully permuted cell layouts. Inputs use per-query causal pool visibility, distinct complete-pool and partial-tail handling, Q8_0 K/V, D=256 and GQA=24/2. Both alternatives compute their own real `TOP_K` from the same supplied indexer scores. Visible scores are shuffled unique integer ranks, exactly representable in F32; their separate RNG is unaffected by the cell permutation. This gives both alternatives the same unambiguous selected pool set while preserving unordered Top-K output. Each case first checks exact selected sets and dense-versus-sparse output error.

The `CROSS` records separate synchronous GGML wall times for:

- `topk_us`: actual Top-K ranking, common to both alternatives.
- `select_us`: compact index generation or the old zero-vector/mask producer, with ranking already available.
- `attention_us`: attention with the selected indices or dense mask already available.
- `qsa_graph_us` and `qsa_p95_us`: the complete score-to-attention graph, including Top-K, selection and attention, measured directly rather than summed from components.
- `compute_buffer` and `selection_bytes`: actual GGML allocator requirement and selection tensor size for that branch. Input storage and private backend pool allocations are excluded; both alternatives coexist, so process VRAM is not an inference peak.

There are 21 samples per stage, five synchronous evaluations per sample, and alternating dense/sparse order. Five warmup evaluations precede each stage. The alternatives have separate first tensor nodes to avoid colliding HIP graph-cache keys; partial/complete stages are measured separately and warmed again. Components run in dependency order, respecting allocator reuse of intermediate storage. After component and complete-graph timings, selection/output bytes must still match the original complete-graph results. Readback and numerical checks are outside timed regions.

The shared Lightning Indexer scoring/projections, RoPE, KV writes, model layers, transfers between devices and DFlash are outside this synthetic timing. It includes the changed selection work and actual Top-K, but cannot establish end-to-end tokens/s, full-model acceptance or real peak VRAM. Keep the existing production-model checks for those claims.

Return the four complete numerical/crossover logs from both GPUs. Use `CROSS_RESULT` (dense median divided by sparse median) together with p95 and both layouts to choose a simple conservative threshold. The current `max(8192, (width <= 8 ? 4 : 16)*n_sel)` rule remains unchanged until those data are available.

## Corrected crossover fixture

The first two user crossover logs from `a1401d2` complete 85 configurations per GPU, then fail the selected-set comparison at 24576 KV cells, width 512, permuted layout. The separate numerical suites complete successfully on both GPUs.

The old fixture drew each visible score independently from 24-bit integers. Replaying its host RNG reproduces one ambiguous cutoff in the failing case: query row 379 has 511 scores greater than 15355057 and two pools (577 and 1301) equal to it. Only one of those pools fits into the 512-pool budget. The existing GPU Top-K radix gather uses an atomic counter for equal-cutoff entries, so either tied pool is a valid result. Independently evaluating Top-K for dense and sparse therefore cannot require identical selected sets for these inputs. The old scalar CPU smoke case did not expose this larger fixture's tie.

The fixture now uses a per-query shuffled permutation of unique F32-exact integer ranks. This removes the ambiguous input from the comparison; it does not change production Top-K, tie handling, QSA selection, attention, or the dispatch threshold. Mask/list comparisons and numerical/component checks remain strict. List mismatch diagnostics now report the device, KV length, width, layout, query row and first differing cell.

Local validation: the test target builds, the existing CPU QSA reference test passes, and the ranked/permuted crossover smoke case still passes. The old failure's score cutoff was reproduced on the host; no GPU rerun is possible in this environment. The successful user numerical runs need not be repeated for this test-only fix. Rebuild `test-qsa-sparse` and rerun only the two `--crossover` commands above. The threshold remains unchanged until both complete with the corrected fixture.
