// Standalone ROCm/HIP reproducer for llama.cpp SDMA faults on file-backed GGUF mmap.
// No llama.cpp headers/libraries required. Build: hipcc -O2 -std=c++17 -pthread rocm_mmap_sdma_repro.cpp -o rocm_mmap_sdma_repro
#include <hip/hip_runtime.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr size_t kTensorBytes = 321126400;  // blk.43.ffn_up_exps.weight: 512 * 627200
constexpr size_t kExpertBytes = 627200;
struct Chunk { size_t offset; size_t bytes; };
// Exactly 59 offset/length pairs from a failing blk.43.ffn_up_exps.weight upload
// trace. Gaps, 512-byte extras and chunk lengths are intentional.
constexpr std::array<Chunk, 59> kChunks = {{
    {0, 10035712}, {10662400, 2509312}, {13798400, 627712},
    {15052800, 4390912}, {20070400, 2509312}, {23206400, 627712},
    {24460800, 6899712}, {33241600, 4390912}, {38259200, 627712},
    {40140800, 1254912}, {42022400, 3136512}, {45785600, 18816512},
    {65228800, 2509312}, {68364800, 6899712}, {75891200, 1882112},
    {78400000, 3136512}, {82790400, 2509312}, {85926400, 6899712},
    {93452800, 5645312}, {100352000, 8781312}, {109760000, 627712},
    {111641600, 627712}, {112896000, 3763712}, {117286400, 6899712},
    {125440000, 5018112}, {131712000, 1882112}, {134848000, 3763712},
    {139238400, 5018112}, {144883200, 5018112}, {150528000, 5645312},
    {156800000, 1882112}, {159308800, 3136512}, {163072000, 16934912},
    {180633600, 5018112}, {186278400, 2509312}, {189414400, 1254912},
    {191923200, 4390912}, {196940800, 2509312}, {200076800, 17562112},
    {218265600, 627712}, {219520000, 1254912}, {221401600, 8154112},
    {230182400, 3763712}, {234572800, 3136512}, {238336000, 6272512},
    {245235200, 3136512}, {248998400, 1254912}, {250880000, 1254912},
    {252761600, 627712}, {254016000, 8781312}, {263424000, 1254912},
    {265305600, 5645312}, {271577600, 3763712}, {275968000, 8154112},
    {284748800, 1882112}, {287257600, 6272512}, {294784000, 19443712},
    {314854400, 5018112}, {320499200, 627200},
}};
static_assert(kChunks.back().offset + kChunks.back().bytes == kTensorBytes);

std::mutex log_mutex;
template <typename... Args>
void logf(const char * fmt, Args... args) {
    char line[1024];
    std::snprintf(line, sizeof(line), fmt, args...);
    std::lock_guard<std::mutex> lock(log_mutex);
    std::cerr << line << std::endl;
}
void hip_check(hipError_t error, const char * expr, int line) {
    if (error != hipSuccess) {
        throw std::runtime_error(std::string("HIP failure at line ") + std::to_string(line) +
                                 " (" + expr + "): " + hipGetErrorString(error));
    }
}
#define HIP_CHECK(expr) hip_check((expr), #expr, __LINE__)

uint64_t parse_number(const std::string & input) {
    if (input.empty() || input[0] == '-') throw std::runtime_error("Invalid number: " + input);
    errno = 0;
    char * end = nullptr;
    const unsigned long long n = std::strtoull(input.c_str(), &end, 0);
    if (errno || end == input.c_str() || *end != 0) throw std::runtime_error("Invalid number: " + input);
    return static_cast<uint64_t>(n);
}
enum class Mode { direct, malloc_staging, pinned_staging };
enum class Sync { each, batch };
enum class Pattern { recorded, rotate, random };
const char * pattern_name(Pattern p) {
    switch (p) {
        case Pattern::recorded: return "recorded";
        case Pattern::rotate: return "rotate";
        case Pattern::random: return "random";
    }
    return "?";
}
uint64_t mix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
const char * mode_name(Mode mode) {
    switch (mode) {
        case Mode::direct: return "direct";
        case Mode::malloc_staging: return "malloc";
        case Mode::pinned_staging: return "pinned";
    }
    return "?";
}
struct Options {
    Mode mode = Mode::direct;
    Sync sync = Sync::each;
    std::vector<int> devices = {0, 1};
    int iterations = 20;
    std::string model_file;
    uint64_t file_offset = 0xd40; // source address modulo 4 KiB in the observed trace
    bool trace = false;
    bool pretouch = false;
    bool map_whole = false;
    Pattern pattern = Pattern::recorded;
    size_t source_span = kTensorBytes;
    uint64_t seed = 0x31c0ffeeULL;
};
// Keep the captured copy sizes/destination offsets, vary just the HIP host sources.
size_t source_offset(const Options & opt, int device, int iteration, size_t index, const Chunk & chunk) {
    if (opt.pattern == Pattern::recorded) return chunk.offset;
    const size_t positions = (opt.source_span - chunk.bytes)/kExpertBytes + 1;
    uint64_t expert;
    if (opt.pattern == Pattern::rotate) {
        expert = chunk.offset/kExpertBytes + static_cast<uint64_t>(iteration)*37 +
                 static_cast<uint64_t>(device)*53;
    } else {
        expert = mix64(opt.seed ^ (static_cast<uint64_t>(device) << 50) ^
                       (static_cast<uint64_t>(iteration) << 20) ^ static_cast<uint64_t>(index));
    }
    return static_cast<size_t>(expert % positions)*kExpertBytes;
}
void usage(const char * program) {
    std::cout << "Usage: " << program << " [options]\n"
              << "  --mode direct|malloc|pinned  HIP source memory (default direct)\n"
              << "  --sync each|batch         Synchronize every copy or after 59 copies (default each)\n"
              << "  --devices 0,1             HIP device indices to use concurrently (default 0,1)\n"
              << "  --iterations N            Complete 59-copy rounds per device (default 20)\n"
              << "  --model-file PATH         Read-only GGUF/blob file instead of sparse temp file\n"
              << "  --map window|whole        mmap only source window or entire file (default window)\n"
              << "  --pattern recorded|rotate|random  Source expert selection (default recorded)\n"
              << "  --source-span-mib N        Source range in MiB starting at --file-offset\n"
              << "                            (default 321126400 bytes; set 1024 for wider churn)\n"
              << "  --seed N                  Random-pattern seed, decimal or 0xhex\n"
              << "  --file-offset N           Byte offset of tensor within file; decimal or 0xhex\n"
              << "                            (default 0xd40 for synthetic file)\n"
              << "  --trace                   Log each upload source range and successful HIP sync\n"
              << "  --pretouch                CPU-read every source page before copies\n"
              << "  --help                    Show help\n";
}
Options parse(int argc, char ** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto value = [&]() -> std::string {
            if (++i == argc) throw std::runtime_error("Missing value for " + arg);
            return argv[i];
        };
        if (arg == "--help") { usage(argv[0]); std::exit(0); }
        else if (arg == "--mode") {
            const auto v = value();
            if (v == "direct") opt.mode = Mode::direct;
            else if (v == "malloc") opt.mode = Mode::malloc_staging;
            else if (v == "pinned") opt.mode = Mode::pinned_staging;
            else throw std::runtime_error("Invalid --mode: " + v);
        } else if (arg == "--sync") {
            const auto v = value();
            if (v == "each") opt.sync = Sync::each;
            else if (v == "batch") opt.sync = Sync::batch;
            else throw std::runtime_error("Invalid --sync: " + v);
        } else if (arg == "--devices") {
            const auto v = value();
            opt.devices.clear();
            size_t pos = 0;
            while (pos < v.size()) {
                const size_t comma = v.find(',', pos);
                const std::string item = v.substr(pos, comma == std::string::npos ? comma : comma-pos);
                const uint64_t n = parse_number(item);
                if (n > static_cast<uint64_t>(std::numeric_limits<int>::max()))
                    throw std::runtime_error("Device index too large");
                const int d = static_cast<int>(n);
                for (int prev : opt.devices) if (prev == d) throw std::runtime_error("Repeated HIP device");
                opt.devices.push_back(d);
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
            if (opt.devices.empty()) throw std::runtime_error("Empty --devices");
        } else if (arg == "--iterations") {
            const uint64_t n = parse_number(value());
            if (!n || n > 1000000) throw std::runtime_error("--iterations out of range");
            opt.iterations = static_cast<int>(n);
        } else if (arg == "--model-file") opt.model_file = value();
        else if (arg == "--map") {
            const auto v = value();
            if (v == "window") opt.map_whole = false;
            else if (v == "whole") opt.map_whole = true;
            else throw std::runtime_error("Invalid --map: " + v);
        }
        else if (arg == "--file-offset") opt.file_offset = parse_number(value());
        else if (arg == "--source-span-mib") {
            const uint64_t mib = parse_number(value());
            if (!mib || mib > SIZE_MAX/(1024ULL*1024ULL))
                throw std::runtime_error("--source-span-mib out of range");
            opt.source_span = static_cast<size_t>(mib * 1024ULL * 1024ULL);
        } else if (arg == "--seed") opt.seed = parse_number(value());
        else if (arg == "--pattern") {
            const auto v = value();
            if (v == "recorded") opt.pattern = Pattern::recorded;
            else if (v == "rotate") opt.pattern = Pattern::rotate;
            else if (v == "random") opt.pattern = Pattern::random;
            else throw std::runtime_error("Invalid --pattern: " + v);
        }
        else if (arg == "--trace") opt.trace = true;
        else if (arg == "--pretouch") opt.pretouch = true;
        else throw std::runtime_error("Unknown option: " + arg);
    }
    return opt;
}

struct SourceMapping {
    int fd = -1;
    void * mapping = MAP_FAILED;
    size_t map_bytes = 0;
    const uint8_t * src = nullptr;
    bool synthetic = false;
    explicit SourceMapping(const Options & opt) {
        const long page = sysconf(_SC_PAGESIZE);
        if (page <= 0) throw std::runtime_error("sysconf page size failed");
        const uint64_t aligned = opt.map_whole ? 0 : opt.file_offset / static_cast<uint64_t>(page) * static_cast<uint64_t>(page);
        const uint64_t delta = opt.file_offset - aligned;
        if (delta > SIZE_MAX - opt.source_span) throw std::runtime_error("Mapping length overflow");
        const size_t window_bytes = static_cast<size_t>(delta) + opt.source_span;
        map_bytes = window_bytes;
        if (aligned > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
            opt.file_offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) - opt.source_span)
            throw std::runtime_error("File offset too large");
        if (opt.model_file.empty()) {
            char tmp[] = "/tmp/rocm-mmap-sdma-XXXXXX";
            fd = mkstemp(tmp);
            if (fd < 0) throw std::runtime_error("mkstemp failed: " + std::string(std::strerror(errno)));
            unlink(tmp);
            synthetic = true;
            // Sparse file: does not write the source span to disk.
            if (ftruncate(fd, static_cast<off_t>(opt.file_offset + opt.source_span)) != 0)
                throw std::runtime_error("ftruncate failed: " + std::string(std::strerror(errno)));
        } else {
            fd = open(opt.model_file.c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error("Cannot open --model-file: " + std::string(std::strerror(errno)));
            struct stat st = {};
            if (fstat(fd, &st) != 0) throw std::runtime_error("fstat failed");
            if (st.st_size < 0 || static_cast<uint64_t>(st.st_size) < opt.file_offset + opt.source_span)
                throw std::runtime_error("Model file too small for offset + source span");
            if (opt.map_whole) {
                if (static_cast<uint64_t>(st.st_size) > SIZE_MAX)
                    throw std::runtime_error("Model file too large to mmap");
                map_bytes = static_cast<size_t>(st.st_size);
            }
        }
        // For the sparse synthetic file, the whole-file and window sizes coincide
        // except for the unaligned file offset. Preserve an exact full-file VMA.
        if (synthetic && opt.map_whole) {
            if (opt.file_offset > SIZE_MAX - opt.source_span)
                throw std::runtime_error("Synthetic mapping overflow");
            map_bytes = static_cast<size_t>(opt.file_offset) + opt.source_span;
        }
        mapping = mmap(nullptr, map_bytes, PROT_READ, MAP_SHARED, fd, static_cast<off_t>(aligned));
        if (mapping == MAP_FAILED) throw std::runtime_error("mmap failed: " + std::string(std::strerror(errno)));
        src = static_cast<const uint8_t *>(mapping) + delta;
    }
    ~SourceMapping() {
        if (mapping != MAP_FAILED) munmap(mapping, map_bytes);
        if (fd >= 0) close(fd);
    }
    SourceMapping(const SourceMapping &) = delete;
    SourceMapping & operator=(const SourceMapping &) = delete;
};

__global__ void probe_kernel(const uint8_t * bytes, size_t length, uint32_t * checksum) {
    const size_t idx = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4096;
    const size_t step = static_cast<size_t>(gridDim.x) * blockDim.x * 4096;
    uint32_t local = 0;
    for (size_t pos = idx; pos < length; pos += step) local += bytes[pos];
    atomicAdd(checksum, local);
}

void stage_free(void * ptr, Mode mode) {
    if (mode == Mode::malloc_staging) std::free(ptr);
    else if (mode == Mode::pinned_staging) HIP_CHECK(hipHostFree(ptr));
}
void worker(int dev, const Options & opt, const uint8_t * source, std::atomic<int> & ready,
            std::atomic<bool> & go) {
    HIP_CHECK(hipSetDevice(dev));
    hipDeviceProp_t props = {};
    HIP_CHECK(hipGetDeviceProperties(&props, dev));
    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    uint8_t * dst = nullptr;
    uint32_t * result = nullptr;
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&dst), kTensorBytes));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&result), sizeof(uint32_t)));
    HIP_CHECK(hipMemsetAsync(dst, 0, kTensorBytes, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    logf("READY dev=%d name=%s dst=%p", dev, props.name, static_cast<void *>(dst));
    ++ready;
    while (!go.load()) std::this_thread::yield();

    for (int iteration = 0; iteration < opt.iterations; ++iteration) {
        std::vector<void *> outstanding; // keep staging buffers alive until stream sync
        if (opt.sync == Sync::batch && opt.mode != Mode::direct) outstanding.reserve(kChunks.size());
        for (size_t i = 0; i < kChunks.size(); ++i) {
            const Chunk chunk = kChunks[i];
            const size_t src_offset = source_offset(opt, dev, iteration, i, chunk);
            const uint8_t * original = source + src_offset;
            const void * hip_source = original;
            void * stage = nullptr;
            if (opt.mode == Mode::malloc_staging) {
                stage = std::malloc(chunk.bytes);
                if (!stage) throw std::runtime_error("malloc staging failed");
            } else if (opt.mode == Mode::pinned_staging) {
                HIP_CHECK(hipHostMalloc(&stage, chunk.bytes));
            }
            if (stage) {
                std::memcpy(stage, original, chunk.bytes);
                hip_source = stage;
            }
            if (opt.trace) {
                logf("COPY BEGIN dev=%d iter=%d chunk=%zu pattern=%s mode=%s src=%p src_end=%p hip_src=%p dst=%p size=%zu src_offset=%zu dst_offset=%zu",
                     dev, iteration, i, pattern_name(opt.pattern), mode_name(opt.mode),
                     static_cast<const void *>(original), static_cast<const void *>(original + chunk.bytes),
                     hip_source, static_cast<void *>(dst + chunk.offset), chunk.bytes, src_offset, chunk.offset);
            }
            HIP_CHECK(hipMemcpyAsync(dst + chunk.offset, hip_source, chunk.bytes, hipMemcpyHostToDevice, stream));
            if (opt.sync == Sync::each) {
                HIP_CHECK(hipStreamSynchronize(stream));
                stage_free(stage, opt.mode);
                if (opt.trace) logf("COPY SYNC END dev=%d iter=%d chunk=%zu", dev, iteration, i);
            } else if (stage) {
                outstanding.push_back(stage);
            }
        }
        if (opt.sync == Sync::batch) {
            HIP_CHECK(hipStreamSynchronize(stream));
            for (void * ptr : outstanding) stage_free(ptr, opt.mode);
        }
        // A small GPU workload after the upload series, like llama.cpp's next graph split.
        HIP_CHECK(hipMemsetAsync(result, 0, sizeof(uint32_t), stream));
        hipLaunchKernelGGL(probe_kernel, dim3(64), dim3(256), 0, stream, dst, kTensorBytes, result);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipStreamSynchronize(stream));
        if (iteration == 0 || (iteration + 1) % 5 == 0 || iteration + 1 == opt.iterations)
            logf("ROUND PASS dev=%d iteration=%d/%d mode=%s sync=%s", dev, iteration+1, opt.iterations,
                 mode_name(opt.mode), opt.sync == Sync::each ? "each" : "batch");
    }
    HIP_CHECK(hipFree(result));
    HIP_CHECK(hipFree(dst));
    HIP_CHECK(hipStreamDestroy(stream));
    uint64_t per_round = 0;
    for (const Chunk & c : kChunks) per_round += c.bytes;
    logf("WORKER PASS dev=%d copies=%llu transferred=%.2f GiB", dev,
         static_cast<unsigned long long>(kChunks.size())*opt.iterations,
         static_cast<double>(per_round)*opt.iterations/(1024.0*1024.0*1024.0));
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const Options opt = parse(argc, argv);
        if (opt.source_span < kTensorBytes)
            throw std::runtime_error("Source span must be at least 321126400 bytes");
        SourceMapping mapping(opt);
        for (const Chunk & c : kChunks) {
            if (c.offset > kTensorBytes || c.bytes > kTensorBytes - c.offset)
                throw std::runtime_error("Invalid built-in chunk bounds");
        }
        int available = 0;
        HIP_CHECK(hipGetDeviceCount(&available));
        for (int dev : opt.devices) {
            if (dev < 0 || dev >= available)
                throw std::runtime_error("Requested HIP device out of range: " + std::to_string(dev));
        }
        if (opt.pretouch) {
            volatile uint8_t sum = 0;
            for (size_t i = 0; i < opt.source_span; i += 4096) sum ^= mapping.src[i];
            sum ^= mapping.src[opt.source_span - 1];
            logf("PRETOUCH DONE checksum=%u", static_cast<unsigned>(sum));
        }
        logf("CONFIG mode=%s sync=%s pattern=%s seed=%llu iterations=%d devices=%zu mapping=%s map_scope=%s file_offset=0x%llx map_bytes=%zu source_span=%zu src=%p chunks=%zu",
             mode_name(opt.mode), opt.sync == Sync::each ? "each" : "batch",
             pattern_name(opt.pattern), static_cast<unsigned long long>(opt.seed), opt.iterations,
             opt.devices.size(), mapping.synthetic ? "synthetic-file-backed" : "model-file-backed",
             opt.map_whole ? "whole" : "window", static_cast<unsigned long long>(opt.file_offset),
             mapping.map_bytes, opt.source_span,
             static_cast<const void *>(mapping.src), kChunks.size());

        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        std::vector<std::thread> threads;
        std::vector<std::exception_ptr> errors(opt.devices.size());
        for (size_t i = 0; i < opt.devices.size(); ++i) {
            threads.emplace_back([&, i]() {
                try { worker(opt.devices[i], opt, mapping.src, ready, go); }
                catch (...) { errors[i] = std::current_exception(); ++ready; }
            });
        }
        while (ready.load() < static_cast<int>(opt.devices.size())) std::this_thread::yield();
        go = true;
        for (auto & thread : threads) thread.join();
        for (const auto & error : errors) if (error) std::rethrow_exception(error);
        logf("TEST PASS: all devices completed %d rounds", opt.iterations);
        return 0;
    } catch (const std::exception & ex) {
        logf("TEST FAIL: %s", ex.what());
        return 1;
    }
}
