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
set -l build build-rocm-moe-stream-reference
set -l fresh
if test -f $build/CMakeCache.txt
    set -l cached_compiler (string match -rg '^CMAKE_HIP_COMPILER:[^=]+=(.*)$' < $build/CMakeCache.txt)
    if test "$cached_compiler" != "$rocm/bin/amdclang++"
        set fresh --fresh
    end
end
cmake -S . -B $build $fresh "-DCMAKE_HIP_COMPILER=$rocm/bin/amdclang++" "-DCMAKE_PREFIX_PATH=$rocm" -C scripts/moe-measure/rocm-sdk-options.cmake -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS='gfx1100;gfx1101' -DLLAMA_MOE_MEASURE=OFF; or exit 1
if not string match -rq '^GGML_HIP:[^=]+=ON$' < $build/CMakeCache.txt
    echo "HIP is disabled in $build" >&2
    exit 1
end
if not string match -rq '^LLAMA_MOE_MEASURE:[^=]+=OFF$' < $build/CMakeCache.txt
    echo "Expected an uninstrumented build: $build" >&2
    exit 1
end
set -l configured_compiler (string match -rg '^CMAKE_HIP_COMPILER:[^=]+=(.*)$' < $build/CMakeCache.txt)
if test "$configured_compiler" != "$rocm/bin/amdclang++"
    echo "Wrong HIP compiler in $build: $configured_compiler" >&2
    exit 1
end
cmake --build $build --target llama-server -j28; or exit 1
python3 scripts/moe-measure/draft-ab.py --repo (pwd) $argv
