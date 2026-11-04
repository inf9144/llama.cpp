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
set -l before build-rocm-moe-before-verify-pair
set -l baseline_id (python3 scripts/moe-measure/build_pair.py build-rocm-moe-stream-reference/bin/llama-server \
    build-rocm-moe-stream-measure/bin/llama-server $before); or exit 1
echo "Baseline: $baseline_id"
for phase in reference measure
    set -l build build-rocm-moe-stream-$phase
    set -l measure OFF
    set -l targets llama-server test-hip-mmvq-pair
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
set -l checks results/moe-verify-pair-check-(date +%Y%m%d-%H%M%S)-$fish_pid
mkdir -p $checks; or exit 1
set -l logs
for backend in ROCm0 ROCm1
    set -l log $checks/pair-$backend.log
    echo "Testing and benchmarking projection pairs on $backend (log: $log)"
    ./build-rocm-moe-stream-reference/bin/test-hip-mmvq-pair --backend $backend --bench > $log 2>&1; or begin
        tail -n 80 $log
        exit 1
    end
    if not string match -rq 'automatic dispatch and guards passed' < $log
        echo "Pair checks were skipped on $backend; inspect $log" >&2
        exit 1
    end
    set -a logs --attach $log
end
set -l diagnosis $checks/completion-3
python3 scripts/moe-measure/completion_diagnose.py --draft-max 3 --production-controls --output $diagnosis --baseline-reference $before/reference/llama-server; or exit 1
python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
checks = list(d["comparisons"].values()) + list(d["replay_comparisons"].values())
if not d["complete"] or not checks or any(not c["tokens_equal"] or c["rows_with_top_id_changes"] or c["max_shared_logprob_delta"] != 0 for c in checks):
    sys.exit("Numerical differences: inspect the diagnosis before benchmarking")
' $diagnosis/report.json; or exit 1
python3 scripts/moe-measure/decode-ab.py --draft-widths 3 --production-controls --profile --output $checks/decode \
    --baseline-reference $before/reference/llama-server \
    --baseline-measure $before/measure/llama-server $logs; or exit 1
python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
pairs = [k for k in d["kernels"] if "mul_mat_vec_q6_k_pair<" in k["name"]]
if len({k["agent"] for k in pairs}) < 2:
    sys.exit("Projection pairs not observed on both GPUs; inspect the profile before claiming a speedup")
print("Production projection pairs:", {k["agent"]: sum(x["calls"] for x in pairs if x["agent"] == k["agent"]) for k in pairs})
' $checks/decode/draft-3/candidate/03-profile/gpu.json
