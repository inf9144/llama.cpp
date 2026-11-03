#!/usr/bin/env fish
# Run from the updated feature checkout, before rebuilding its existing binaries.
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
set -l before build-rocm-moe-before-expert-group
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

echo "Checking HIP headers from $rocm"
printf '#include <hip/hip_bf16.h>\n' | $rocm/bin/amdclang++ --rocm-path=$rocm --hip-path=$rocm -I$rocm/include --offload-arch=gfx1100 -D__HIP_PLATFORM_AMD__=1 -std=c++17 -x hip -fsyntax-only -; or exit 1
for phase in reference measure
    set -l build build-rocm-moe-stream-$phase
    set -l measure OFF
    set -l targets llama-server test-hip-mmvq-moe test-backend-ops
    if test $phase = measure
        set measure ON
        set targets llama-server
    end
    set -l fresh
    if test -f $build/CMakeCache.txt
        set -l cached_compiler (string match -rg '^CMAKE_HIP_COMPILER:[^=]+=(.*)$' < $build/CMakeCache.txt)
        if test "$cached_compiler" != "$rocm/bin/amdclang++"
            echo "Reinitializing CMake metadata for $build (compiler changed)"
            set fresh --fresh
        end
    end
    cmake -S . -B $build $fresh "-DCMAKE_HIP_COMPILER=$rocm/bin/amdclang++" "-DCMAKE_PREFIX_PATH=$rocm" -C scripts/moe-measure/rocm-sdk-options.cmake -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS='gfx1100;gfx1101' -DLLAMA_MOE_MEASURE=$measure -DLLAMA_BUILD_TESTS=ON; or exit 1
    if not string match -rq '^GGML_HIP:[^=]+=ON$' < $build/CMakeCache.txt
        echo "HIP is disabled in $build after configuration; stopping before build" >&2
        exit 1
    end
    if not string match -rq "^LLAMA_MOE_MEASURE:[^=]+=$measure\$" < $build/CMakeCache.txt
        echo "Wrong instrumentation mode in $build; stopping before build" >&2
        exit 1
    end
    set -l configured_compiler (string match -rg '^CMAKE_HIP_COMPILER:[^=]+=(.*)$' < $build/CMakeCache.txt)
    if test "$configured_compiler" != "$rocm/bin/amdclang++"
        echo "Wrong HIP compiler in $build: $configured_compiler" >&2
        exit 1
    end
    cmake --build $build --target $targets -j28; or exit 1
end
set -l logs
for backend in ROCm0 ROCm1
    set -l native moe-grouped-$backend.log
    set -l ops moe-grouped-ops-$backend.log
    echo "Testing and benchmarking $backend (logs: $native, $ops)"
    ./build-rocm-moe-stream-reference/bin/test-hip-mmvq-moe --backend $backend --bench > $native 2>&1; or begin
        tail -n 60 $native
        exit 1
    end
    ./build-rocm-moe-stream-reference/bin/test-backend-ops -b $backend -o MUL_MAT_ID_CACHE_GLU > $ops 2>&1; or begin
        tail -n 60 $ops
        exit 1
    end
    set -a logs --attach $native --attach $ops
end
set -l diagnosis results/moe-expert-group-check-(date +%Y%m%d-%H%M%S)
python3 scripts/moe-measure/completion_diagnose.py --draft-max 3 --output $diagnosis --baseline-reference $before/reference/llama-server; or exit 1
python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
checks = list(d["comparisons"].values()) + list(d["replay_comparisons"].values())
if not d["complete"] or not checks or any(not c["tokens_equal"] or c["rows_with_top_id_changes"] or c["max_shared_logprob_delta"] != 0 for c in checks):
    sys.exit("Numerical differences: inspect the diagnosis before benchmarking")
' $diagnosis/report.json; or exit 1
python3 scripts/moe-measure/decode-ab.py --draft-widths 3 --profile --compare-sampling \
    --baseline-reference $before/reference/llama-server \
    --baseline-measure $before/measure/llama-server $logs
