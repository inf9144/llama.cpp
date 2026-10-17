# Text allocation plan recovery

The graph allocator keeps one allocation plan. A graph using raw or mixed embeddings can require different backend copies and replace the initial worst-case text plan with a plan for the current batch. After that, alternating MTP catch-up batches without outputs and single-token drafting batches can repeatedly invalidate each other's tensor allocation limits, even when the backend buffers are large enough.

Each llama context marks its text plan for recovery after non-text graphs or a scheduler reallocation callback. On the next ordinary text microbatch it synchronizes outstanding work and reserves the worst-case text graph using the existing scheduler and allocator. It does not apply the simulated full memory context. Old graph results are invalidated before the real batch is built and bound again.

The reservation uses the same input, output and architecture-specific limits as the final startup reservation. In particular, the main context can require more verification outputs than the MTP draft length when N-Gram speculation is also enabled. The draft context retains its own output limit. Stable text execution adds no recurring reservation or synchronization.

The normal allocator validity checks and synchronization fallback remain in place. A failed recovery returns an allocation error and leaves recovery pending. No model arithmetic, speculative algorithm, KV/recurrent state serialization or cache format changes are involved.

## CPU regression tests

```sh
cmake -S . -B build-recovery -DLLAMA_BUILD_TESTS=ON -DGGML_VULKAN=OFF -DGGML_HIP=OFF
cmake --build build-recovery --target test-alloc test-llama-archs test-batch-alloc -j 2
ctest --test-dir build-recovery -R '^(test-alloc|test-graph-recovery|test-batch-alloc)$' --output-on-failure
```

`test-alloc` exercises same-count allocation-plan oscillation, recovery, the 49-output case and buffer retention with two buffer types, including scheduler configurations with one or multiple pipeline copies. `test-graph-recovery` uses small CPU Llama, Qwen35 and Qwen35-MTP models generated in memory, with no model download. It runs with F16 KV and with Q8_0 KV plus Flash Attention. It checks repeated raw/mixed embedding transitions, alternating output counts, checkpoint restoration into an independent context, equal continuation logits, scheduler identity, buffer size and graph reuse.

## Local GPU validation

Use the companion diagnostic branch for the detailed measurements. A short prompt followed by a small image and text generation should be sufficient for an initial check; the first test need not fill a 75k-token context.

- Build and run Vulkan with both AMD GPUs and the existing layer split.
- Repeat with ROCm/HIP where the devices and build are supported.
- Check short text-only inference, then text/image/text with MTP and N-Gram enabled.
- Repeat several images, including images spanning multiple microbatches.
- Exercise longer N-Gram verification batches as well as one-to-three MTP draft tokens.
- Continue through KV padding boundaries and long-context generation.
- Restore checkpoints and RAM/SSD prompt caches, then compare continuation behavior.
- Record reallocation counts, recovery counts and timings, including synchronization in explicit reservation.
- Compare device buffer sizes, peak VRAM and host/pinned memory before and after recovery.
- Compare server timings and end-to-end tokens/s with matching requests and warm-up.
- Test configured pipeline parallelism separately from the single-copy production setup.

Required graph transitions may still replan. After text recovery, draft/catch-up alternation should stop causing persistent size invalidations. Recovery uses the original reservation bounds; it is not a promise that every architecture or future graph variant fits the same plan.

Synchronization time can include already pending GPU work. A reduction in reallocation synchronization alone is not an equal end-to-end speedup: count the recovery path too. Vulkan and ROCm runtime validation requires actual GPU hardware.
