#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __HIPCC__
#define QSA_TEST_HD __host__ __device__
#else
#define QSA_TEST_HD
#endif

namespace qsa_test {

QSA_TEST_HD inline int bound(const int32_t * values, int begin, int end, int32_t value, bool upper) {
    while (begin < end) {
        const int mid = begin + (end - begin)/2;
        if (values[mid] < value || (upper && values[mid] == value)) {
            begin = mid + 1;
        } else {
            end = mid;
        }
    }
    return begin;
}

QSA_TEST_HD inline void swap_ordered(int32_t & a, int32_t & b) {
    if (a > b) {
        const int32_t t = a;
        a = b;
        b = t;
    }
}

QSA_TEST_HD inline void sort4(int32_t & a, int32_t & b, int32_t & c, int32_t & d) {
    swap_ordered(a, b);
    swap_ordered(c, d);
    swap_ordered(a, c);
    swap_ordered(b, d);
    swap_ordered(b, c);
}

#ifdef GGML_TEST_HIP_QSA_SELECT
int gpu_suite(const char * backend_name);
#endif

} // namespace qsa_test

#undef QSA_TEST_HD
