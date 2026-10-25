# ROCm SDMA file-backed mmap reproducer

Standalone HIP experiment for the multi-GPU SDMA host-read fault observed in
`inf9144/llama.cpp`'s `diagnostics/rocm-checkpoint-fault` branch.
**No llama.cpp libraries, models, scheduler or cache infrastructure are needed to compile this.**

## Observed symptom / motivation

- Host OS: CachyOS kernel `7.3.0-5-cachyos` (in-tree amdgpu), ROCm/HIP `7.2.4`
- GPUs: RX 7900 XTX + RX 7800 XT, both in one ROCm process
- Failure: HSA reports `Memory access fault ... Page not present or supervisor privilege`
  on a **GPU SDMA0 read** from a valid, file-backed, `r--s` model mmap.
- llama.cpp uploads from `blk.43.ffn_up_exps.weight`, especially the original
  offset `294784000`, length `19443712` bytes, consistently encounter faults.
- Explicit CPU-prefaulting and stream synchronization did not prevent faults.
- Changing **only** the HIP source pointer from the original mmap to a temporary
  `malloc` buffer containing the same bytes was sufficient to complete a full
  model prompt in the original application (also worked with pinned staging).
- Relevant upstream tracker:
  https://github.com/ROCm/rocm-systems/issues/4817
  (previous KFD range-splitting bug fixed in kernel 7.2.10 / 7.3-rc4;
  this occurrence on kernel 7.3 may be related, but is NOT proven identical).

This standalone test **attempts** to isolate that failure. A clean run does not
disprove the bug: llama.cpp's full scheduling, allocation and reuse patterns might
be required to trigger it.

## Build (Arch/CachyOS, fish)

Run from this directory:

```fish
/opt/rocm/bin/hipcc -O2 -std=c++17 -pthread rocm_mmap_sdma_repro.cpp -o rocm_mmap_sdma_repro
```

If hipcc is on PATH, `hipcc` works too. You must have the HIP SDK/runtime
and a working ROCm driver; build with the actual installed ROCm toolchain.

## First run: synthetic file-backed MAP_SHARED mapping

A temporary **sparse regular file** (~306 MiB logical size) is created and
immediately unlinked. It is mmap'ed `PROT_READ | MAP_SHARED`, and accessed at
offset `0xd40`, which recreates the non-page-aligned source address. This
does not write a 306-MiB file full of data.

```fish
set -e AMD_SERIALIZE_KERNEL
set -e AMD_SERIALIZE_COPY
./rocm_mmap_sdma_repro --devices 0,1 --iterations 20 --mode direct --sync each --trace 2>&1 | tee rocm-repro-direct.log
```

A test that does **not** fault is still useful. A sparse zero-filled file may
have different filesystem and page-cache behavior from the real GGUF.

## Strongest test: original GGUF blob + exact tensor file offset

The failing source pointer was verified against `/proc/<pid>/maps` to be
inside a shared read-only Hugging Face GGUF blob, not inside the process heap.
From that mapping and the captured copy offsets, the **file-relative** start of
the 321126400-byte `blk.43.ffn_up_exps.weight` tensor is:

`0x7c2331d40`  (tensor offset in that exact GGUF blob).

For the model build in the investigation, the blob was:

```text
/home/hausen/.cache/huggingface/hub/models--ukisai--Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/blobs/70ce6fb48a94fe34bd7e9a53a526be68066c64b5a5077cdc82fb31350b66b608
```

The exact file offset is model/blob-version specific. Check the source file has
not changed; with a different quantization or blob, recalculate the tensor
offset rather than silently using this one.

```fish
set gguf /home/hausen/.cache/huggingface/hub/models--ukisai--Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/blobs/70ce6fb48a94fe34bd7e9a53a526be68066c64b5a5077cdc82fb31350b66b608
./rocm_mmap_sdma_repro --model-file $gguf --file-offset 0x7c2331d40 --devices 0,1 --iterations 50 --mode direct --sync each --trace 2>&1 | tee rocm-gguf-direct.log
```

By default the program maps only the 306-MiB tensor window, **not** the entire
GGUF file. That differs from the original llama.cpp process, which held a
multi-GiB file-backed VMA. To test whether the SVM mapping extent matters,
add `--map whole` (this maps the **entire file as virtual address space**,
not as fully resident RAM). With `--map window` the earlier behavior is
preserved. Compare otherwise identical runs:

```fish
./rocm_mmap_sdma_repro --model-file $gguf --file-offset 0x7c2331d40 --map whole --devices 0,1 --iterations 50 --mode direct --sync each --trace 2>&1 | tee rocm-gguf-whole-direct.log
./rocm_mmap_sdma_repro --model-file $gguf --file-offset 0x7c2331d40 --map whole --devices 0,1 --iterations 50 --mode malloc --sync each --trace 2>&1 | tee rocm-gguf-whole-malloc.log
```

The test uses the same 59 offsets/sizes observed in one failing llama.cpp
run, **not** its dynamically changing expert selections over all inference
steps.
Copies go to `device_buffer + offset` on each device. The two HIP devices run
concurrently in separate host threads. After each 59-copy round, a small HIP
kernel reads from the GPU buffer.

## Varying expert ranges / higher upload pressure

The captured 59 H2D uploads were a **single snapshot** of one layer, whereas
llama.cpp repeatedly changes expert selections, uploads gate/up/down weights,
and copies experts from many layers. The original 50-round test repeated the
*same* source offsets every round (~265 MiB per GPU per round; 25.9 GiB across
two GPUs for 50 rounds).

Use the new source-range controls to stress registration/unpinning churn:

- `--pattern recorded` (default): exact original 59 **source offsets**.
- `--pattern rotate`: deterministically shift 59 source ranges between
  rounds and GPUs, preserving recorded lengths and GPU destination offsets.
- `--pattern random`: independently choose new aligned expert-sized source
  offsets for each chunk/round/GPU; deterministic with `--seed`.
- `--source-span-mib 1024`: select from a 1-GiB source window starting at
  `--file-offset`; default is the original 321126400-byte tensor span.
  Values larger than the tensor may touch **other GGUF tensors**. This is a
  deliberate HIP memory-traffic stress test, **not exact MoE semantics**.

Both GPUs share the original read-only file-backed mmap but use separate HIP
streams and device buffers. Source offsets always remain within the mapped
window; recorded destination offsets and transfer sizes do not change.

Suggested tests (using `$gguf` from above):

```fish
./rocm_mmap_sdma_repro --model-file $gguf --file-offset 0x7c2331d40 --map whole --pattern rotate --devices 0,1 --iterations 250 --mode direct --sync each 2>&1 | tee rocm-rotate-direct.log

./rocm_mmap_sdma_repro --model-file $gguf --file-offset 0x7c2331d40 --map whole --source-span-mib 1024 --pattern random --seed 20261025 --devices 0,1 --iterations 250 --mode direct --sync each 2>&1 | tee rocm-random-direct.log
```

250 rounds represent about 129.4 GiB of aggregate H2D requests to two
devices, without accounting for kernel reads or HIP internal staging.
`--trace` can be added for precise source ranges but produces many log lines.
For a comparable malloc control, rerun the **identical command** replacing
only `--mode direct` with `--mode malloc`.

If changing offsets still does not reproduce, compare `--sync batch` vs.
`--sync each`, then consider recording/replaying the actual per-layer
scheduler upload events. This synthetic stress test deliberately does **not**
reproduce the real MoE cache's SLRU eviction pattern or all GGUF tensors.

## A/B comparisons

Repeat with **identical file, file offset, devices, iterations, sync mode**,
changing only `--mode`:

```fish
./rocm_mmap_sdma_repro --model-file $gguf --file-offset 0x7c2331d40 --devices 0,1 --iterations 50 --mode malloc --sync each --trace 2>&1 | tee rocm-gguf-malloc.log

./rocm_mmap_sdma_repro --model-file $gguf --file-offset 0x7c2331d40 --devices 0,1 --iterations 50 --mode pinned --sync each --trace 2>&1 | tee rocm-gguf-pinned.log
```

- `direct`: `hipMemcpyAsync(device, file_mmap_pointer, ...)`.
- `malloc`: CPU `memcpy` of the exact bytes into a fresh pageable allocation,
  then `hipMemcpyAsync(device, malloc_pointer, ...)`.
- `pinned`: same except staging memory is allocated with `hipHostMalloc`.
- `--sync each`: `hipStreamSynchronize` after **every** upload, before freeing
  each staging allocation, mirroring the successful app A/B control.
- `--sync batch`: enqueue all 59 transfers on each device stream and synchronize
  once; staging allocations remain **alive** until that synchronization.
- `--pretouch`: optional CPU read of each source page before upload.
- `--trace`: print the original source pointer, actual HIP source pointer, range,
  and completed sync; logs may become large.

If `--sync each` does not reproduce, try `--sync batch` with larger iteration
counts, and compare `--devices 0` vs `--devices 0,1`. The `batch` variant
changes the amount of overlapping HIP work and is **not** otherwise identical
to the `each` run. Never compare differently configured workloads as an
equivalent A/B result.

**Warning:** Faulting GPU processes may be terminated, reset the ROCm context,
and leave device state unhealthy until the process and/or driver recover.
Do not run with valuable unsaved GPU work in progress.

## What to collect for an upstream report

1. Full commands and logs from `direct` and control runs.
2. `uname -a`, `hipconfig --version`, `pacman -Q rocm-core hip-runtime-amd`,
   `modinfo -n amdgpu`, and GPU IDs / PCI addresses.
3. `sudo journalctl -k -b --no-pager | grep -E 'amdgpu|kfd|SDMA|page fault'`
   captured after a crash (remove unrelated log entries if needed).
4. The printed `src` and `src_end` range for the upload whose address contains
   the HSA fault.
5. Whether `direct` faults while `malloc` or `pinned` completes on **the same**
   kernel, ROCm and model file.

The goal is to determine whether the fault exists in standalone HIP, and only
then narrow ROCclr/ROCR/AMDKFD ownership. It is **not** established that the
previously fixed KFD bug is present again.
