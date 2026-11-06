"""Exercise the production host pool with deferred copies, without claiming HIP validation."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which('g++'), 'g++ unavailable')
class StagePoolTests(unittest.TestCase):
    def test_deferred_source_lifetimes_budget_and_order(self):
        source = (ROOT / 'ggml/src/ggml-cuda/ggml-cuda.cu').read_text()
        pool = source[source.index('struct ggml_rocm_malloc_stage_slot {'):source.index('#endif // defined(GGML_USE_HIP)', source.index('struct ggml_rocm_malloc_stage_slot {'))]
        harness = r'''
#include "ggml-moe.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>
#undef GGML_ASSERT
#define GGML_ASSERT(x) assert(x)
#define GGML_LOG_WARN(...) ((void) 0)
#define CUDA_CHECK(x) assert((x) == 0)
using cudaStream_t = int;
struct event { int stream = 0; size_t through = 0; };
using cudaEvent_t = event *;
using hipError_t = int;
constexpr int hipSuccess=0, hipErrorNotReady=1, hipHostMallocDefault=0;
constexpr int cudaEventDisableTiming=0, cudaMemcpyHostToDevice=0;
std::vector<std::function<void()>> queues[2];
size_t completed[2] = {}, records = 0, queries = 0, synchronizes = 0, pin_attempts = 0, frees = 0;
bool fail_pin = false, graph_waiting = false;
void run(int stream, size_t through) { while (completed[stream] < through) queues[stream][completed[stream]++](); }
int hipHostMalloc(void ** p, size_t n, int) { ++pin_attempts; if (fail_pin) return 2; *p = std::malloc(n); assert(*p); return 0; }
int cudaFreeHost(void * p) { assert(!graph_waiting && "hipHostFree would wait for the blocked graph"); ++frees; std::free(p); return 0; }
int cudaEventCreateWithFlags(event ** p, int) { *p = new event; return 0; }
int cudaEventDestroy(event * p) { assert(completed[p->stream] >= p->through); delete p; return 0; }
int hipEventQuery(event * p) { ++queries; return completed[p->stream] >= p->through ? 0 : 1; }
int cudaEventSynchronize(event * p) { ++synchronizes; run(p->stream, p->through); return 0; }
int cudaEventRecord(event * p, int stream) { ++records; p->stream = stream; p->through = queues[stream].size(); return 0; }
int cudaMemcpyAsync(void * dst, const void * src, size_t n, int, int stream) {
    queues[stream].push_back([=] { std::memcpy(dst, src, n); }); return 0;
}
int hipGetLastError() { return 0; }
const char * hipGetErrorString(int) { return "pin failure"; }
int ggml_cuda_env_int(const char *, int value) { return value; }
'''
        cases = r'''
int main() {
    constexpr size_t limit = 256*1024;
    {
        // Fill the old pool with small slots, then request a larger miss batch.
        // Freeing any pinned slot here would wait for the graph waiting for us.
        ggml_rocm_malloc_stage_pool pool;
        ggml_rocm_malloc_stage_config cfg{true, limit, ""};
        std::vector<unsigned char> input(limit, 7), output(limit, 0xa5);
        ggml_tensor tensor = {}; tensor.data = output.data();
        for (int i = 0; i < 4; ++i) pool.copy(0, output.data() + 17*i, input.data(), 17, limit);
        assert(pool.slots.size() == 4);
        for (const auto & slot : pool.slots) assert(slot.capacity == 65536);
        run(0, queues[0].size());
        const size_t pinned_before = pin_attempts, freed_before = frees;
        const size_t reserved_before = pool.reserved_bytes;
        const ggml_moe_upload batch[] = {
            {&tensor, input.data(), 0, 65536},
            {&tensor, input.data(), 65536, 65536},
            {&tensor, input.data(), 131072, 65536},
        };
        graph_waiting = true;
        pool.copy_batch(1, batch, 3, cfg);
        std::fill(input.begin(), input.end(), 0xff);
        run(1, queues[1].size());
        assert(pin_attempts == pinned_before && frees == freed_before);
        assert(pool.reserved_bytes == reserved_before && pool.reserved_bytes <= limit);
        for (size_t i = 0; i < output.size(); ++i) assert(output[i] == (i < 196608 ? 7 : 0xa5));
        graph_waiting = false;
    }
    {
        // Small pending transfers must not consume eight 64-MiB slots and stall.
        ggml_rocm_malloc_stage_pool pool;
        constexpr size_t budget = 512*1024*1024;
        unsigned char input[17];
        std::fill(std::begin(input), std::end(input), 23);
        std::vector<unsigned char> output(16*sizeof(input), 0);
        const size_t waited_before = synchronizes;
        for (int i = 0; i < 16; ++i) pool.copy(0, output.data() + i*sizeof(input), input, sizeof(input), budget);
        assert(pool.reserved_bytes == 16*65536);
        assert(synchronizes == waited_before);
        std::fill(std::begin(input), std::end(input), 0xff);
        run(0, queues[0].size());
        assert(std::all_of(output.begin(), output.end(), [](unsigned char x) { return x == 23; }));
    }
    {
        // A large single range must obey the same chunk bound as a batch.
        constexpr size_t chunk = 64*1024*1024, bytes = chunk + 17;
        ggml_rocm_malloc_stage_pool pool;
        ggml_rocm_malloc_stage_config cfg{true, chunk + 65536, ""};
        std::vector<unsigned char> input(bytes, 13), output(bytes + 38, 0xa5);
        ggml_tensor tensor = {}; tensor.data = output.data();
        const ggml_moe_upload upload{&tensor, input.data(), 19, bytes};
        pool.copy_batch(0, &upload, 1, cfg);
        assert(pool.slots.size() == 2 && pool.slots[0].capacity == chunk && pool.slots[1].capacity == 65536);
        assert(pool.reserved_bytes <= cfg.limit);
        std::fill(input.begin(), input.end(), 0xff);
        run(0, queues[0].size());
        for (size_t i = 0; i < output.size(); ++i) assert(output[i] == (i >= 19 && i < bytes + 19 ? 13 : 0xa5));
    }
    for (bool pin_failure : {false, true}) for (const char * filter : {"", "ffn_", "excluded"}) {
        fail_pin = pin_failure;
        ggml_rocm_malloc_stage_config cfg{true, limit, filter};
        ggml_rocm_malloc_stage_pool pool;
        std::vector<unsigned char> dest[3], source[3], expected[3];
        ggml_tensor tensors[3] = {};
        for (int i = 0; i < 3; ++i) {
            dest[i].resize(4*limit); source[i].resize(4*limit); expected[i].resize(4*limit);
            tensors[i].data = dest[i].data();
            std::strcpy(tensors[i].name, i == 1 ? "metadata" : "ffn_weights");
        }
        for (int iteration = 0; iteration < 4; ++iteration) {
            std::vector<ggml_moe_upload> uploads;
            for (int i = 0; i < 3; ++i) {
                std::fill(dest[i].begin(), dest[i].end(), 0xa5);
                std::fill(expected[i].begin(), expected[i].end(), 0xa5);
                for (size_t j = 0; j < source[i].size(); ++j) source[i][j] = (j*13 + iteration*31)%251;
                size_t at = 19;
                for (size_t bytes : {size_t(0), size_t(17), size_t(65535), size_t(65536), size_t(65537), size_t(2*limit + 3)}) {
                    uploads.push_back({tensors+i, source[i].data()+at, at, bytes});
                    std::memcpy(expected[i].data()+at, source[i].data()+at, bytes);
                    at += bytes + 31;
                }
            }
            uploads.push_back({tensors, source[0].data()+500, 20, 31});
            std::memcpy(expected[0].data()+20, source[0].data()+500, 31);
            const int stream = iteration%2;
            pool.copy_batch(stream, uploads.data(), uploads.size(), cfg);
            assert(pool.reserved_bytes <= limit);
            for (int i = 0; i < 3; ++i) if (cfg.matches({tensors+i, nullptr, 0, 1})) std::fill(source[i].begin(), source[i].end(), 0xff);
            run(stream, queues[stream].size());
            for (int i = 0; i < 3; ++i) assert(dest[i] == expected[i]);
        }
        // Reuse the same pool on both streams without a completion between calls.
        const ggml_moe_upload small[] = {{tensors, source[0].data(), 0, 16}, {tensors+2, source[2].data(), 0, 17}};
        pool.copy_batch(0, small, 2, cfg); pool.copy_batch(1, small, 2, cfg);
        for (int s=0;s<2;++s) run(s, queues[s].size());
    }
    fail_pin = false;
    {
        std::vector<unsigned char> input(4096, 7), output(8192, 0);
        ggml_tensor tensor = {}; tensor.data = output.data();
        {
            ggml_rocm_malloc_stage_pool pool;
            ggml_rocm_malloc_stage_config cfg{true, limit, ""};
            const ggml_moe_upload a{&tensor, input.data(), 0, 4096}, b{&tensor, input.data(), 4096, 4096};
            pool.copy_batch(0, &a, 1, cfg); pool.copy_batch(1, &b, 1, cfg);
            std::fill(input.begin(), input.end(), 0xff);
        }
        assert(std::all_of(output.begin(), output.end(), [](unsigned char x){return x==7;}));
    }
    {
        ggml_rocm_malloc_stage_pool pool;
        ggml_rocm_malloc_stage_config cfg{true, limit, ""};
        std::vector<unsigned char> input(4096, 3), output(4096*12, 0);
        ggml_tensor tensor = {}; tensor.data = output.data();
        std::vector<ggml_moe_upload> uploads;
        for (int i=0;i<12;++i) uploads.push_back({&tensor, input.data(), size_t(i)*4096, 4096});
        const size_t before = records;
        pool.copy_batch(0, uploads.data(), uploads.size(), cfg);
        assert(records == before + 1);
        std::fill(input.begin(),input.end(),0xff);
        run(0, queues[0].size());
        assert(std::all_of(output.begin(),output.end(),[](unsigned char x){return x==3;}));
        pool.copy_batch(1, nullptr, 0, cfg);
        assert(records == before + 1);
    }
    puts("Production stage pool: no frees under graph waits, deferred copies, bounded reuse, mixed filters, pin fallback, oversized ranges, two streams, guards and event batching passed");
}
'''
        with tempfile.TemporaryDirectory() as directory:
            cpp = Path(directory) / 'pool.cpp'
            binary = Path(directory) / 'pool'
            cpp.write_text(harness + pool + cases)
            build = subprocess.run(['g++', '-std=c++17', '-O1', '-g', '-fsanitize=address,undefined',
                                    '-fno-omit-frame-pointer', '-I', str(ROOT / 'ggml/src'),
                                    '-I', str(ROOT / 'ggml/include'), str(cpp), '-o', str(binary)],
                                   text=True, capture_output=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            environment = os.environ.copy()
            # LeakSanitizer cannot enumerate /proc tasks in the tool sandbox.
            environment['ASAN_OPTIONS'] = 'detect_leaks=0'
            result = subprocess.run([str(binary)], text=True, capture_output=True, timeout=60, env=environment)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('event batching passed', result.stdout)


if __name__ == '__main__':
    unittest.main()
