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
set -lx GGML_ROCM_MALLOC_STAGE_POOL 1
set -lx GGML_ROCM_MALLOC_STAGE_POOL_MIB 512
set -lx GGML_ROCM_MALLOC_STAGE_POOL_TENSOR ''
set -l before build-rocm-moe-before-selector-overlap
set -l baseline_id (python3 scripts/moe-measure/build_pair.py build-rocm-moe-stream-reference/bin/llama-server \
    build-rocm-moe-stream-measure/bin/llama-server $before); or exit 1
echo "Baseline: $baseline_id"
for phase in reference measure
    set -l build build-rocm-moe-stream-$phase
    set -l measure OFF
    set -l targets llama-server test-llama-archs test-moe-stream
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
    cmake -S . -B $build $fresh "-DCMAKE_HIP_COMPILER=$rocm/bin/amdclang++" "-DCMAKE_PREFIX_PATH=$rocm" \
        -C scripts/moe-measure/rocm-sdk-options.cmake -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON \
        -DGPU_TARGETS='gfx1100;gfx1101' -DLLAMA_MOE_MEASURE=$measure -DLLAMA_BUILD_TESTS=ON; or exit 1
    if not string match -rq '^GGML_HIP:[^=]+=ON$' < $build/CMakeCache.txt
        echo "HIP is disabled in $build" >&2
        exit 1
    end
    cmake --build $build --target $targets -j28; or exit 1
end
set -l candidate_id (python3 scripts/moe-measure/build_pair.py build-rocm-moe-stream-reference/bin/llama-server \
    build-rocm-moe-stream-measure/bin/llama-server); or exit 1
echo "Candidate: $candidate_id"
if test "$candidate_id" = "$baseline_id"
    echo 'Baseline and candidate are identical; the baseline must be saved before rebuilding.' >&2
    exit 1
end
set -l checks results/moe-selector-overlap-check-(date +%Y%m%d-%H%M%S)-$fish_pid
mkdir -p $checks; or exit 1
set -l logs
for backend in ROCm0 ROCm1
    for check in selector overlap cache
        set -l log $checks/$check-$backend.log
        echo "Testing $check on $backend (log: $log)"
        set -l binary ./build-rocm-moe-stream-reference/bin/test-llama-archs
        set -l args --backend $backend -s 42
        switch $check
            case selector
                set -a args --test-dflash-selector
            case overlap
                set binary ./build-rocm-moe-stream-reference/bin/test-moe-stream
                set args --backend $backend --overlap-only
            case cache
                set -a args --test-moe-cache-prefill
        end
        $binary $args > $log 2>&1
        set -l check_status $status
        if test $check_status -ne 0
            tail -n 60 $log
            exit $check_status
        end
        if test $check = overlap; and not string match -rq 'device lookup:.*split=1.*12 graph calls' < $log
            echo "Split lookup checks were skipped on $backend; inspect $log" >&2
            exit 1
        end
        if test $check = cache; and not string match -rq 'split requests=[1-9][0-9]*' < $log
            echo "Shared-expert overlap was not active on $backend; inspect $log" >&2
            exit 1
        end
        set -a logs --attach $log
    end
end
for width in 0 3
    set -l diagnosis $checks/completion-$width
    python3 scripts/moe-measure/completion_diagnose.py --draft-max $width --production-controls \
        --output $diagnosis --baseline-reference $before/reference/llama-server; or exit 1
    python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
checks = list(d["comparisons"].values()) + list(d["replay_comparisons"].values())
if not d["complete"] or not checks or any(not c["tokens_equal"] or c["rows_with_top_id_changes"] or c["max_shared_logprob_delta"] != 0 for c in checks):
    sys.exit("Numerical differences: inspect the diagnosis before benchmarking")
' $diagnosis/report.json; or exit 1
end
python3 scripts/moe-measure/decode-ab.py --draft-widths 3 --production-controls --profile --output $checks/decode \
    --baseline-reference $before/reference/llama-server --baseline-measure $before/measure/llama-server $logs; or exit 1
python3 -c '
import collections, json, sys
d = json.load(open(sys.argv[1]))
split = collections.Counter()
for x in d["kernels"]:
    if "finish_lookup<true>" in x["name"]:
        split[x["agent"]] += x["calls"]
print("Production split lookups:", dict(split))
if len(split) < 2:
    sys.exit("Split lookup not observed on both GPUs; inspect the candidate profile")
' $checks/decode/draft-3/candidate/03-profile/gpu.json; or exit 1
