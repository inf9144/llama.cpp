#include "qsa-select-common.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// Check the rank calculation shared by the merge and tail-merge kernels against std::sort.
static void test_merge() {
    std::mt19937 rng(0x515341);
    int checked = 0;
    for (int size : {256, 512, 1024, 2048, 4096}) {
        for (int pattern = 0; pattern < 32; ++pattern) {
            std::vector<int32_t> original(size);
            for (int i = 0; i < size; ++i) {
                original[i] = pattern == 0 || rng()%5 == 0 ? INT32_MAX : int32_t(rng()%(pattern%2 ? 17 : 262144));
            }
            auto expected = original;
            std::sort(expected.begin(), expected.end());
            auto input = original;
            std::vector<int32_t> output(size);
            for (int i = 0; i < size; i += 4) {
                qsa_test::sort4(input[i], input[i + 1], input[i + 2], input[i + 3]);
            }
            for (int run = 4; run < size; run *= 2) {
                std::vector<int> writes(size, 0);
                for (int i = 0; i < size; ++i) {
                    const int group = i/(2*run)*(2*run);
                    const bool right = i - group >= run;
                    const int other = group + (right ? 0 : run);
                    const int rank = qsa_test::bound(input.data(), other, other + run, input[i], right) - other;
                    const int dst = group + (i - group)%run + rank;
                    GGML_ASSERT(dst >= 0 && dst < size && writes[dst]++ == 0);
                    output[dst] = input[i];
                }
                input.swap(output);
            }
            GGML_ASSERT(input == expected);
            std::array<int32_t, 3> tail = {int32_t(rng()%17), int32_t(rng()%17), INT32_MAX};
            if (pattern%3 == 0) { tail.fill(INT32_MAX); }
            std::sort(tail.begin(), tail.end());
            std::vector<int32_t> merged(size + 3);
            std::vector<int> writes(size + 3, 0);
            for (int i = 0; i < size; ++i) {
                const int rank = std::lower_bound(tail.begin(), tail.end(), input[i]) - tail.begin();
                GGML_ASSERT(writes[i + rank]++ == 0);
                merged[i + rank] = input[i];
            }
            for (int i = 0; i < 3; ++i) {
                const int rank = qsa_test::bound(input.data(), 0, size, tail[i], true);
                GGML_ASSERT(writes[rank + i]++ == 0);
                merged[rank + i] = tail[i];
            }
            expected.insert(expected.end(), tail.begin(), tail.end());
            std::sort(expected.begin(), expected.end());
            GGML_ASSERT(merged == expected);
            ++checked;
        }
    }
    printf("QSA selection: %d pool/tail merge cases match std::sort (duplicates and invalid padding included)\n", checked);
}

int main(int argc, char ** argv) {
    test_merge();
    if (argc == 1) {
        printf("GPU candidate comparison: use --backend ROCm0 or --backend ROCm1\n");
        return 0;
    }
    if (argc != 3 || strcmp(argv[1], "--backend") != 0) {
        fprintf(stderr, "usage: %s [--backend ROCm0|ROCm1]\n", argv[0]);
        return 1;
    }
#ifdef GGML_TEST_HIP_QSA_SELECT
    return qsa_test::gpu_suite(argv[2]);
#else
    fprintf(stderr, "GPU candidates require a build with GGML_HIP=ON; no GPU results produced.\n");
    return 1;
#endif
}
