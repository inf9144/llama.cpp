#!/usr/bin/env fish
# Preserve the current binaries before rebuilding the updated feature checkout.
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
set -l before build-rocm-moe-before-miss-handoff
if test -x $before/reference/llama-server; and test -x $before/measure/llama-server
    echo "Reusing baseline: $before"
else
    test ! -e $before; or exit 1
    test -x build-rocm-moe-stream-reference/bin/llama-server; or exit 1
    test -x build-rocm-moe-stream-measure/bin/llama-server; or exit 1
    mkdir $before; or exit 1
    cp -a build-rocm-moe-stream-reference/bin $before/reference; or exit 1
    cp -a build-rocm-moe-stream-measure/bin $before/measure; or exit 1
end
for phase in reference measure
    set -l build build-rocm-moe-stream-$phase
    set -l measure OFF
    set -l targets llama-server test-moe-stream test-llama-archs
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
set -l checks results/moe-miss-check-(date +%Y%m%d-%H%M%S)-$fish_pid
mkdir -p $checks; or exit 1
set -l logs
./build-rocm-moe-stream-reference/bin/test-llama-archs --test-moe-cache-policy -s 42 > $checks/policy.log 2>&1; or begin
    cat $checks/policy.log
    exit 1
end
set -a logs --attach $checks/policy.log
for backend in ROCm0 ROCm1
    set -l log $checks/lookup-$backend.log
    echo "Testing cache-map updates on $backend (log: $log)"
    ./build-rocm-moe-stream-reference/bin/test-moe-stream --backend $backend --lookup-only > $log 2>&1; or begin
        tail -n 80 $log
        exit 1
    end
    if not string match -rq 'sparse map updates: 512 changes.*passed' < $log
        echo "Sparse map checks were skipped on $backend; inspect $log" >&2
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
python3 scripts/moe-measure/decode-ab.py --draft-widths 3 --production-controls --profile \
    --baseline-reference $before/reference/llama-server \
    --baseline-measure $before/measure/llama-server $logs
