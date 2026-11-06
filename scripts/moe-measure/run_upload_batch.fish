#!/usr/bin/env fish
cd (realpath (status dirname)/../..); or exit 1
if test (git branch --show-current) != feature/moe-cache-prefill
    echo 'Expected feature/moe-cache-prefill' >&2
    exit 1
end
set -l rocm $HOME/opt/rocm-10.1
if not test -x $rocm/bin/amdclang++; or not test -d $rocm/include/hip
    echo "Expected ROCm 10.1 SDK at $rocm" >&2
    exit 1
end
set -lx ROCM_PATH $rocm
set -lx HIP_PATH $rocm
set -lx PATH $rocm/bin $PATH
set -lx LD_LIBRARY_PATH $rocm/lib $LD_LIBRARY_PATH
set -l before build-rocm-moe-before-upload-batch
set -l baseline_id (python3 scripts/moe-measure/build_pair.py build-rocm-moe-stream-reference/bin/llama-server \
    build-rocm-moe-stream-measure/bin/llama-server $before); or exit 1
echo "Baseline: $baseline_id"
for phase in reference measure
    set -l build build-rocm-moe-stream-$phase
    set -l measure OFF
    set -l targets llama-server test-moe-stream
    if test $phase = measure
        set measure ON
        set targets llama-server
    end
    set -l fresh
    if test -f $build/CMakeCache.txt
        set -l cached_compiler (string match -rg '^CMAKE_HIP_COMPILER:[^=]+=(.*)$' < $build/CMakeCache.txt)
        if test "$cached_compiler" != "$rocm/bin/amdclang++"
            set fresh --fresh
        end
    end
    cmake -S . -B $build $fresh "-DCMAKE_HIP_COMPILER=$rocm/bin/amdclang++" "-DCMAKE_PREFIX_PATH=$rocm" -C scripts/moe-measure/rocm-sdk-options.cmake -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS='gfx1100;gfx1101' -DLLAMA_MOE_MEASURE=$measure -DLLAMA_BUILD_TESTS=ON; or exit 1
    if not string match -rq '^GGML_HIP:[^=]+=ON$' < $build/CMakeCache.txt
        echo "HIP is disabled in $build" >&2
        exit 1
    end
    if not string match -rq "^LLAMA_MOE_MEASURE:[^=]+=$measure\$" < $build/CMakeCache.txt
        echo "Wrong instrumentation mode in $build" >&2
        exit 1
    end
    set -l configured_compiler (string match -rg '^CMAKE_HIP_COMPILER:[^=]+=(.*)$' < $build/CMakeCache.txt)
    if test "$configured_compiler" != "$rocm/bin/amdclang++"
        echo "Wrong HIP compiler in $build: $configured_compiler" >&2
        exit 1
    end
    cmake --build $build --target $targets -j28; or exit 1
end
set -l candidate_id (python3 scripts/moe-measure/build_pair.py build-rocm-moe-stream-reference/bin/llama-server \
    build-rocm-moe-stream-measure/bin/llama-server); or exit 1
echo "Candidate: $candidate_id"
if test "$candidate_id" = "$baseline_id"
    echo 'Baseline and candidate are the same runtime build. Use a snapshot saved before rebuilding the candidate.' >&2
    exit 1
end
set -l checks results/moe-upload-batch-check-(date +%Y%m%d-%H%M%S)-$fish_pid
mkdir -p $checks; or exit 1
set -l logs
set -lx GGML_ROCM_MALLOC_STAGE_POOL 1
set -lx GGML_ROCM_MALLOC_STAGE_POOL_MIB 512
set -lx GGML_ROCM_MALLOC_STAGE_POOL_TENSOR ''
for backend in ROCm0 ROCm1
    set -l log $checks/upload-$backend.log
    echo "Testing uploads and cache lookup on $backend (log: $log)"
    echo "Native upload checks: backend=$backend" > $log
    for check in upload-only upload-bench lookup-only
        set -l pool_mib 512
        if test $check = upload-only
            set pool_mib 1
        end
        echo "Running $backend --$check (pool=$pool_mib MiB)"
        echo "BEGIN backend=$backend test=$check pool_mib=$pool_mib" >> $log
        env GGML_ROCM_MALLOC_STAGE_POOL_MIB=$pool_mib ./build-rocm-moe-stream-reference/bin/test-moe-stream --backend $backend --$check >> $log 2>&1
        set -l check_status $status
        echo "END backend=$backend test=$check exit=$check_status" >> $log
        if test $check_status -ne 0
            echo "Native test failed: backend=$backend test=$check exit=$check_status (log: $log)" >&2
            tail -n 80 $log
            exit $check_status
        end
    end
    if not string match -rq 'upload batch: byte equality.*passed' < $log
        echo "Upload checks were skipped on $backend; inspect $log" >&2
        exit 1
    end
    if not string match -rq 'sparse map updates: 512 changes.*passed' < $log
        echo "Lookup checks were skipped on $backend; inspect $log" >&2
        exit 1
    end
    set -a logs --attach $log
end
for width in 0 3
    set -l diagnosis $checks/completion-$width
    python3 scripts/moe-measure/completion_diagnose.py --draft-max $width --production-controls --output $diagnosis --baseline-reference $before/reference/llama-server; or exit 1
    python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
checks = list(d["comparisons"].values()) + list(d["replay_comparisons"].values())
if not d["complete"] or not checks or any(not c["tokens_equal"] or c["rows_with_top_id_changes"] or c["max_shared_logprob_delta"] != 0 for c in checks):
    sys.exit("Numerical differences: inspect the diagnosis before benchmarking")
' $diagnosis/report.json; or exit 1
end
python3 scripts/moe-measure/decode-ab.py --draft-widths 3 --production-controls --profile --output $checks/decode \
    --baseline-reference $before/reference/llama-server \
    --baseline-measure $before/measure/llama-server $logs; or exit 1
python3 -c '
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
layers = [x for phase in ("cold", "warm") for x in json.load(open(p / f"host-{phase}.json"))["layers"]]
misses = {x["backend"] for x in layers if x.get("device_lookup") and x.get("device_miss")}
batched = {x["backend"] for x in layers if x.get("upload_batches", 0)}
if not misses or misses - batched:
    sys.exit("Batched device-miss uploads not observed on the expected GPUs; inspect the profile")
print("Production miss uploads:", sum(x.get("upload_batches", 0) for x in layers), "batches,",
      sum(x.get("upload_copies", 0) for x in layers), "ranges")
' $checks/decode/draft-3/candidate/03-profile
