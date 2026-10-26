#if defined(GGML_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
#include "common.cuh"
#include "mmb.cuh"
#include "mmid.cuh"
#include "unary.cuh"

// BF16 WMMA prefill GEMM adapted from stew675/llama.cpp (0d30433).
namespace {
typedef float v8f  __attribute__((ext_vector_type(8)));

// Fragment-layout shim.  Both gens compute a 16x16x16 tile with fp32 accumulation; only the lane
// bookkeeping differs (gated_delta_net_chunked_bf16.cu vs ..._gfx11.cu):
//   gfx11: 16 halfs/lane (the whole K row), accumulator m = 2*e + hi
//   gfx12:  8 halfs/lane ("two runs of four": k = 4*hi + {0..3} and 4*hi + 8 + {0..3}), m = 8*hi + e
typedef short mmb_frag_t      __attribute__((ext_vector_type(16)));
typedef short mmb_bf16_frag_t __attribute__((ext_vector_type(16)));
typedef short mmb_f16_frag_t  __attribute__((ext_vector_type(16)));
#define MMB_ACC_M(e, hi) (2*(e) + (hi))
// gfx11 keeps the whole 16-half row in one lane.  A macro (not a function) so the preprocessed
// source at every call site is the original expression -- the gfx11 asm stays bit-identical.
#define mmb_ld_frag(p, hi) (__builtin_bit_cast(mmb_frag_t, (uint4[2]){*reinterpret_cast<const uint4 *>(p), *reinterpret_cast<const uint4 *>((p) + 8)}))

constexpr int MMB_BK = 64, MMB_NT = 256, MMB_LDS_STRIDE = MMB_BK + 8;
__device__ __forceinline__ uint16_t mmb_f2bf(float f) { uint32_t u = __float_as_uint(f); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
// RNE float->bf16.  `v_cvt_pk_bf16_f32` does not exist on gfx11, and `__bf16` lowers to a
// truncating convert (10/4096 values differ from RNE), so the rounding is done in integer and the
// pair is assembled with `v_perm_b32` (bytes 2,3 of each rounded word) -- one instruction where the
// shift/or form needed two, and no `>>16` per value.  Selector 0x03020706 = {ua.b2, ua.b3, ub.b2,
// ub.b3}: v_perm's byte 0..3 come from its second operand, 4..7 from its first.  Bit-identical
// (incl. +0/-0, `d == 0` weights) -- verified 1M random pairs, `PPL`/greedy/width gates.
// The shift/or form.  Kept for the IQ3 dequants: they are LUT-latency bound and the v_perm form,
// which needs all eight rounded words before it can start, is measurably slower there
// (IQ3_S GLU 1318 -> 1355 ms without DBUF, 1225 -> 1258 with).  Every other dequant prefers perm.
__device__ __forceinline__ uint32_t mmb_pack2_so(float a, float b) { return (uint32_t)mmb_f2bf(a) | ((uint32_t)mmb_f2bf(b) << 16); }

__device__ __forceinline__ uint32_t mmb_pack2(float a, float b) {
    uint32_t ua = __float_as_uint(a), ub = __float_as_uint(b);
    ua += 0x7fffu + ((ua >> 16) & 1u);
    ub += 0x7fffu + ((ub >> 16) & 1u);
    return __builtin_amdgcn_perm(ua, ub, 0x03020706u);
}
__constant__ int8_t mmb_kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__device__ __forceinline__ float mmb_h2f(uint16_t h) { return (float) __builtin_bit_cast(_Float16, h); }

// Portable WMMA dispatch: gfx12 renamed the builtins and uses the 8-half fragment above; gfx11 keeps
// the first-gen builtin and its 16-half fragment.  The kernels are runtime-gated to RDNA3_5/RDNA4
// (mmb_enabled), so only the matching arm is ever launched.
__device__ __forceinline__ v8f mmb_wmma_bf16(mmb_frag_t a, mmb_frag_t b, v8f c) {
#if defined(RDNA3)
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#else
    return c;
#endif
}
__device__ __forceinline__ v8f mmb_wmma_f16(mmb_frag_t a, mmb_frag_t b, v8f c) {
#if defined(RDNA3)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    return c;
#endif
}

__global__ void mmb_cvt_f32_bf16(const float * __restrict__ x, uint16_t * __restrict__ y, const size_t n) {
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i + 8 <= n) {
        const float4 a = *(const float4 *)(x + i), b = *(const float4 *)(x + i + 4);
        uint4 o; o.x = mmb_pack2(a.x, a.y); o.y = mmb_pack2(a.z, a.w); o.z = mmb_pack2(b.x, b.y); o.w = mmb_pack2(b.z, b.w);
        *(uint4 *)(y + i) = o;
    } else {
        for (; i < n; ++i) y[i] = mmb_f2bf(x[i]);
    }
}

// dequantize one weight row's 64-value quarter of an IQ4_XS super-block (136 B: half d, uint16
// scales_h, uint8 scales_l[4], qs[128]).  Same nibble LUT as IQ4_NL; the 6-bit sub-block scale is
// split scale = (scales_l[q4] nibble) | (scales_h bits 4*q4 .. +1) << 4, then value = d*(scale-32)*kv.
__device__ __forceinline__ void mmb_dq_row_iq4xs(const uint8_t * base, const int q4, uint32_t * arow) {
    const float dh = mmb_h2f(*(const uint16_t *) base);
    const uint32_t sh = *(const uint16_t *)(base + 2);
    const uint32_t sl = *(const uint8_t  *)(base + 4 + q4);
    const uint32_t * qA = (const uint32_t *)(base + 8 + 32 * q4);
    const uint32_t * qB = (const uint32_t *)(base + 8 + 32 * q4 + 16);
    const int sA = (int)((sl & 0xf) | (((sh >> (4 * q4))     & 3) << 4));
    const int sB = (int)((sl >> 4)  | (((sh >> (4 * q4 + 2)) & 3) << 4));
    const float dA = dh * (float)(sA - 32), dB = dh * (float)(sB - 32);
    const uint32_t L0 = 0x3f2d1801u, L1 = 0x766a5d4fu, L2 = 0xa6998d81u, L3 = 0xf1d9c5b5u;
#pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
        const float d = blk ? dB : dA; const float md = -128.0f * d; const uint32_t * q = blk ? qB : qA; uint32_t * out = arow + blk * 16;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t v = q[w];
            const uint32_t nib[2] = { v & 0x0F0F0F0Fu, (v >> 4) & 0x0F0F0F0Fu };
            float x[2][4];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint32_t n = nib[h];
                const uint32_t sel = n & 0x07070707u;
                const uint32_t pA = __builtin_amdgcn_perm(L1, L0, sel);
                const uint32_t pB = __builtin_amdgcn_perm(L3, L2, sel);
                const uint32_t m  = ((n >> 3) & 0x01010101u) * 0xFFu;
                const uint32_t u  = (pA & ~m) | (pB & m);
                x[h][0] = fmaf((float)(u & 0xFFu), d, md);
                x[h][1] = fmaf((float)((u >> 8) & 0xFFu), d, md);
                x[h][2] = fmaf((float)((u >> 16) & 0xFFu), d, md);
                x[h][3] = fmaf((float)(u >> 24), d, md);
            }
            out[2*w]         = mmb_pack2(x[0][0], x[0][1]); out[2*w + 1]     = mmb_pack2(x[0][2], x[0][3]);
            out[8 + 2*w]     = mmb_pack2(x[1][0], x[1][1]); out[8 + 2*w + 1] = mmb_pack2(x[1][2], x[1][3]);
        }
    }
}

// dequantize one weight row's 64-value quarter of a Q3_K super-block (110 B: hmask[32], qs[64],
// scales[12], half d) into 64 bf16.  Quarter q4: half n = q4>>1; two 32-value groups at j =
// 2*(q4&1)+jj, 2-bit field shift 2*j, high bit hmask bit (4n+j); 16-value sub-blocks is0 carry the
// 6-bit scale.  value = d*(scale-32) * ((qs>>shift)&3 - (high ? 0 : 4)).
__device__ __forceinline__ void mmb_dq_row_q3k(const uint8_t * base, const int q4, uint32_t * arow) {
    const uint8_t * hm = base;
    const uint8_t * qn = base + 32 + 32 * (q4 >> 1);
    const uint8_t * sc = base + 96;
    const float d = mmb_h2f(*(const uint16_t *)(base + 108));
    const int n = q4 >> 1;
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
        const int j = 2 * (q4 & 1) + jj;
        const int shift = 2 * j;
        const uint32_t m = 1u << (4 * n + j);
        uint32_t * out = arow + 16 * jj;
#pragma unroll
        for (int is0 = 0; is0 < 2; ++is0) {
            const int is = 8 * n + 2 * j + is0;
            int us;
            if (is < 4)       us = (sc[is] & 0xF) | (((sc[is + 8] >> 0) & 3) << 4);
            else if (is < 8)  us = (sc[is] & 0xF) | (((sc[is + 4] >> 2) & 3) << 4);
            else if (is < 12) us = (sc[is - 8] >> 4) | (((sc[is]     >> 4) & 3) << 4);
            else              us = (sc[is - 8] >> 4) | (((sc[is - 4] >> 6) & 3) << 4);
            const float dl = d * (float)(us - 32);
#pragma unroll
            for (int l = 0; l < 16; l += 2) {
                const int li = 16 * is0 + l;
                const float v0 = dl * (float)((int)((qn[li]     >> shift) & 3) - ((hm[li]     & m) ? 0 : 4));
                const float v1 = dl * (float)((int)((qn[li + 1] >> shift) & 3) - ((hm[li + 1] & m) ? 0 : 4));
                out[li / 2] = mmb_pack2(v0, v1);
            }
        }
    }
}

// dequantize one weight row's two consecutive IQ4_NL blocks (36 bytes) into 64 bf16 in LDS.
// LUT held in registers as (kv + 128) bytes and applied with v_perm_b32 (4 nibbles per op pair) instead of a per-lane
// indexed constant array (which lowers to one scalar-byte memory load per element). The value kv*d is produced as
// fma(kv+128, d, -128*d): -128*d is exact, so the single rounding equals RN(kv*d) -> bitwise the same BF16 as before.
__device__ __forceinline__ void mmb_dq_row36(const uint4 w0, const uint4 w1, const uint32_t w2, uint32_t * arow) {
    const uint32_t ws[9] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w, w2};
    const float d0 = mmb_h2f((uint16_t)(ws[0] & 0xffff)), d1 = mmb_h2f((uint16_t)(ws[4] >> 16));
    const uint32_t q0[4] = { (ws[0] >> 16) | (ws[1] << 16), (ws[1] >> 16) | (ws[2] << 16), (ws[2] >> 16) | (ws[3] << 16), (ws[3] >> 16) | (ws[4] << 16) };
    const uint32_t q1[4] = { ws[5], ws[6], ws[7], ws[8] };
    // kv + 128 = {1,24,45,63,79,93,106,118,129,141,153,166,181,197,217,241} packed little-endian, 4 per dword
    const uint32_t L0 = 0x3f2d1801u, L1 = 0x766a5d4fu, L2 = 0xa6998d81u, L3 = 0xf1d9c5b5u;
#pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
        const float d = blk ? d1 : d0; const float md = -128.0f * d; const uint32_t * q = blk ? q1 : q0; uint32_t * out = arow + blk * 16;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t v = q[w];
            const uint32_t nib[2] = { v & 0x0F0F0F0Fu, (v >> 4) & 0x0F0F0F0Fu };
            float x[2][4];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint32_t n = nib[h];
                const uint32_t sel = n & 0x07070707u;
                const uint32_t pA = __builtin_amdgcn_perm(L1, L0, sel);   // entries 0..7
                const uint32_t pB = __builtin_amdgcn_perm(L3, L2, sel);   // entries 8..15
                const uint32_t m  = ((n >> 3) & 0x01010101u) * 0xFFu;      // 0xFF where the nibble >= 8
                const uint32_t u  = (pA & ~m) | (pB & m);
                x[h][0] = fmaf((float)(u & 0xFFu), d, md);
                x[h][1] = fmaf((float)((u >> 8) & 0xFFu), d, md);
                x[h][2] = fmaf((float)((u >> 16) & 0xFFu), d, md);
                x[h][3] = fmaf((float)(u >> 24), d, md);
            }
            out[2*w] = mmb_pack2(x[0][0], x[0][1]); out[2*w + 1] = mmb_pack2(x[0][2], x[0][3]);
            out[8 + 2*w] = mmb_pack2(x[1][0], x[1][1]); out[8 + 2*w + 1] = mmb_pack2(x[1][2], x[1][3]);
        }
    }
}

// dequantize one weight row's two consecutive Q8_0 blocks (68 bytes: d0 qs0[32] d1 qs1[32]) into 64 bf16 in LDS
__device__ __forceinline__ void mmb_dq_row68(const uint4 w0, const uint4 w1, const uint4 w2, const uint4 w3, const uint32_t w4, uint32_t * arow) {
    const uint32_t ws[17] = {w0.x,w0.y,w0.z,w0.w, w1.x,w1.y,w1.z,w1.w, w2.x,w2.y,w2.z,w2.w, w3.x,w3.y,w3.z,w3.w, w4};
    const float d0 = mmb_h2f((uint16_t)(ws[0] & 0xffff)), d1 = mmb_h2f((uint16_t)(ws[8] >> 16));
#pragma unroll
    for (int blk = 0; blk < 2; ++blk) {
        const float d = blk ? d1 : d0; uint32_t * out = arow + blk * 16;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const uint32_t v = blk ? ws[9 + w] : ((ws[w] >> 16) | (ws[w + 1] << 16));
            const float e0 = d * (float)(int8_t)(v      ), e1 = d * (float)(int8_t)(v >>  8);
            const float e2 = d * (float)(int8_t)(v >> 16), e3 = d * (float)(int8_t)(v >> 24);
            out[2*w] = mmb_pack2(e0, e1); out[2*w + 1] = mmb_pack2(e2, e3);
        }
    }
}

// dequantize one weight row's 64-value quarter of a Q4_K super-block. A 256-value super-block is
// 144 B: half2 dm at +0, scales[12] at +4, qs[128] at +16; the quarter (ksh & 3) takes qs 32 bytes
// at +16 + 32*(ksh&3). Low nibbles -> values 0..31, high nibbles -> 32..63; sub-block scales are
// is = 2*(ksh&3) and is+1 via get_scale_min_k4. On-the-fly so a 120 GB model needs no bf16 shadow.
__device__ __forceinline__ void mmb_dq_row_q4k(const uint4 hdr, const uint4 q0, const uint4 q1, const int ksh, uint32_t * arow) {
    const float dall = mmb_h2f((uint16_t)(hdr.x & 0xffff));
    const float dmin = mmb_h2f((uint16_t)(hdr.x >> 16));
    auto sbyte = [&](const int i) -> uint32_t {
        const uint32_t w = (i < 4) ? hdr.y : (i < 8) ? hdr.z : hdr.w;
        return (w >> (8 * (i & 3))) & 0xFFu;
    };
    auto gsm = [&](const int j, float & dd, float & mm) {
        const uint32_t q0b = sbyte(j), q4b = sbyte(j + 4);
        uint32_t sc, mi;
        if (j < 4) { sc = q0b & 63u; mi = q4b & 63u; }
        else       { sc = (q4b & 0xFu) | ((sbyte(j - 4) >> 6) << 4); mi = (q4b >> 4) | ((q0b >> 6) << 4); }
        dd = dall * (float) sc; mm = dmin * (float) mi;
    };
    const int is = 2 * (ksh & 3);
    float d1, m1, d2, m2; gsm(is, d1, m1); gsm(is + 1, d2, m2);
    const uint32_t qs[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        const uint32_t v = qs[w];
        const uint32_t lo = v & 0x0F0F0F0Fu, hi = (v >> 4) & 0x0F0F0F0Fu;
        arow[2*w]             = mmb_pack2(fmaf((float)( lo        & 0xFFu), d1, -m1), fmaf((float)((lo >>  8) & 0xFFu), d1, -m1));
        arow[2*w + 1]         = mmb_pack2(fmaf((float)((lo >> 16) & 0xFFu), d1, -m1), fmaf((float)((lo >> 24)      ), d1, -m1));
        arow[16 + 2*w]        = mmb_pack2(fmaf((float)( hi        & 0xFFu), d2, -m2), fmaf((float)((hi >>  8) & 0xFFu), d2, -m2));
        arow[16 + 2*w + 1]    = mmb_pack2(fmaf((float)((hi >> 16) & 0xFFu), d2, -m2), fmaf((float)((hi >> 24)      ), d2, -m2));
    }
}

// dequantize one weight row's 64-value quarter of an IQ3_S super-block (110 B: half d, qs[64],
// qh[8], signs[32], scales[4]) into 64 bf16. q4 = ksh & 3 selects the 64-value quarter
// (groups ib = 2*q4, 2*q4+1). Fields are non-contiguous, so this loads straight from the row base
// (called from store_lds, not the register prefetch).  value = d*(1+2*scale_nibble) * grid[3-bit]*sign.
__device__ __forceinline__ void mmb_dq_row_iq3s(const uint8_t * base, const int q4, uint32_t * arow) {
    const float d = mmb_h2f(*(const uint16_t *) base);
    const uint8_t sc = *(const uint8_t *)(base + 106 + q4);
    const uint8_t * qs = base + 2 + 16 * q4;
    const uint8_t * qh = base + 66 + 2 * q4;
    const uint8_t * sg = base + 74 + 8 * q4;
#pragma unroll
    for (int g = 0; g < 2; ++g) {
        const float dg = d * (float)(1 + 2 * ((sc >> (4 * g)) & 0xf));
        uint32_t * out = arow + 16 * g;
        const uint32_t qhg = qh[g];
#pragma unroll
        for (int il = 0; il < 4; ++il) {
            const uint32_t grid1 = iq3s_grid[qs[8*g + 2*il    ] | ((qhg << (8 - 2*il)) & 256)];
            const uint32_t grid2 = iq3s_grid[qs[8*g + 2*il + 1] | ((qhg << (7 - 2*il)) & 256)];
            const uint32_t signs = sg[4*g + il];
            float v[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float g1 = dg * (float)((grid1 >> (8*j)) & 0xFFu);
                const float g2 = dg * (float)((grid2 >> (8*j)) & 0xFFu);
                v[j]     = (signs >> j)     & 1u ? -g1 : g1;
                v[4 + j] = (signs >> (4+j)) & 1u ? -g2 : g2;
            }
            out[4*il + 0] = mmb_pack2_so(v[0], v[1]);
            out[4*il + 1] = mmb_pack2_so(v[2], v[3]);
            out[4*il + 2] = mmb_pack2_so(v[4], v[5]);
            out[4*il + 3] = mmb_pack2_so(v[6], v[7]);
        }
    }
}

// PARTS-way split of the above across the threads that share one A row, so that a BM < MMB_NT tile
// (the GLU's BM=64) does not run the A dequant on only MMB_NT/BM of the warps.  Thread `p` takes the
// `il` groups {p, p+PARTS, ...} for both g, writing the same LDS dwords as the whole-row form.
template <int PARTS>
__device__ __forceinline__ void mmb_dq_row_iq3s_p(const uint8_t * base, const int q4, uint32_t * arow, const int p) {
    const float d = mmb_h2f(*(const uint16_t *) base);
    const uint8_t sc = *(const uint8_t *)(base + 106 + q4);
    const uint8_t * qs = base + 2 + 16 * q4;
    const uint8_t * qh = base + 66 + 2 * q4;
    const uint8_t * sg = base + 74 + 8 * q4;
#pragma unroll
    for (int g = 0; g < 2; ++g) {
        const float dg = d * (float)(1 + 2 * ((sc >> (4 * g)) & 0xf));
        uint32_t * out = arow + 16 * g;
        const uint32_t qhg = qh[g];
#pragma unroll
        for (int il = p; il < 4; il += PARTS) {
            const uint32_t grid1 = iq3s_grid[qs[8*g + 2*il    ] | ((qhg << (8 - 2*il)) & 256)];
            const uint32_t grid2 = iq3s_grid[qs[8*g + 2*il + 1] | ((qhg << (7 - 2*il)) & 256)];
            const uint32_t signs = sg[4*g + il];
            float v[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float g1 = dg * (float)((grid1 >> (8*j)) & 0xFFu);
                const float g2 = dg * (float)((grid2 >> (8*j)) & 0xFFu);
                v[j]     = (signs >> j)     & 1u ? -g1 : g1;
                v[4 + j] = (signs >> (4+j)) & 1u ? -g2 : g2;
            }
            out[4*il + 0] = mmb_pack2_so(v[0], v[1]);
            out[4*il + 1] = mmb_pack2_so(v[2], v[3]);
            out[4*il + 2] = mmb_pack2_so(v[4], v[5]);
            out[4*il + 3] = mmb_pack2_so(v[6], v[7]);
        }
    }
}

// Register-resident form of the split dequant above.  `mmb_iq3s_preload` runs inside the
// `load_regs` prefetch (which overlaps the previous K-step's WMMA) and `mmb_dq_iq3s_r` then works
// purely out of registers, so the dequant phase that runs exposed between the WMMA phases issues no
// global loads at all.  Only the `il` groups this thread owns are fetched (4/PARTS of them), and the
// two `qs` bytes a grid lookup needs are adjacent, so they come back as one uint16.
template <int PARTS>
struct mmb_iq3s_r {
    uint32_t d, sc, qh;
    uint16_t qs[2][4 / PARTS];
    uint8_t  sg[2][4 / PARTS];
};

template <int PARTS>
__device__ __forceinline__ mmb_iq3s_r<PARTS> mmb_iq3s_preload(const uint8_t * base, const int q4, const int p) {
    mmb_iq3s_r<PARTS> r;
    r.d  = *(const uint16_t *) base;
    r.sc = *(const uint8_t *)(base + 106 + q4);
    r.qh = *(const uint16_t *)(base + 66 + 2 * q4);
    const uint8_t * qs = base + 2 + 16 * q4;
    const uint8_t * sg = base + 74 + 8 * q4;
#pragma unroll
    for (int g = 0; g < 2; ++g)
#pragma unroll
        for (int it = 0; it < 4 / PARTS; ++it) {
            const int il = p + it * PARTS;
            r.qs[g][it] = *(const uint16_t *)(qs + 8 * g + 2 * il);
            r.sg[g][it] = sg[4 * g + il];
        }
    return r;
}

template <int PARTS>
__device__ __forceinline__ void mmb_dq_iq3s_r(const mmb_iq3s_r<PARTS> & R, uint32_t * arow, const int p) {
    const float d = mmb_h2f((uint16_t) R.d);
    const uint8_t sc = (uint8_t) R.sc;
#pragma unroll
    for (int g = 0; g < 2; ++g) {
        const float dg = d * (float)(1 + 2 * ((sc >> (4 * g)) & 0xf));
        uint32_t * out = arow + 16 * g;
        const uint32_t qhg = (R.qh >> (8 * g)) & 0xff;
#pragma unroll
        for (int it = 0; it < 4 / PARTS; ++it) {
            const int il = p + it * PARTS;
            const uint32_t qs2 = R.qs[g][it];
            const uint32_t grid1 = iq3s_grid[(qs2 & 0xffu)     | ((qhg << (8 - 2*il)) & 256)];
            const uint32_t grid2 = iq3s_grid[(qs2 >> 8)        | ((qhg << (7 - 2*il)) & 256)];
            const uint32_t signs = R.sg[g][it];
            float v[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float g1 = dg * (float)((grid1 >> (8*j)) & 0xFFu);
                const float g2 = dg * (float)((grid2 >> (8*j)) & 0xFFu);
                v[j]     = (signs >> j)     & 1u ? -g1 : g1;
                v[4 + j] = (signs >> (4+j)) & 1u ? -g2 : g2;
            }
            out[4*il + 0] = mmb_pack2_so(v[0], v[1]);
            out[4*il + 1] = mmb_pack2_so(v[2], v[3]);
            out[4*il + 2] = mmb_pack2_so(v[4], v[5]);
            out[4*il + 3] = mmb_pack2_so(v[6], v[7]);
        }
    }
}

// dequantize one weight row's 64-value quarter of a Q5_K super-block (176 B: half2 dm, scales[12],
// qh[32], qs[128]) into 64 bf16.  Same affine form as Q4_K plus the 5th bit from qh (bit 2*q4 for
// the low nibbles, 2*q4+1 for the high nibbles).  Loaded from the row base in store_lds.
__device__ __forceinline__ void mmb_dq_row_q5k(const uint8_t * base, const int q4, uint32_t * arow) {
    const uint4 hdr = *(const uint4 *) base;
    const float dall = mmb_h2f((uint16_t)(hdr.x & 0xffff));
    const float dmin = mmb_h2f((uint16_t)(hdr.x >> 16));
    auto sbyte = [&](const int i) -> uint32_t {
        const uint32_t w = (i < 4) ? hdr.y : (i < 8) ? hdr.z : hdr.w;
        return (w >> (8 * (i & 3))) & 0xFFu;
    };
    auto gsm = [&](const int j, float & dd, float & mm) {
        const uint32_t q0b = sbyte(j), q4b = sbyte(j + 4);
        uint32_t sc, mi;
        if (j < 4) { sc = q0b & 63u; mi = q4b & 63u; }
        else       { sc = (q4b & 0xFu) | ((sbyte(j - 4) >> 6) << 4); mi = (q4b >> 4) | ((q0b >> 6) << 4); }
        dd = dall * (float) sc; mm = dmin * (float) mi;
    };
    const int is = 2 * q4;
    float d1, m1, d2, m2; gsm(is, d1, m1); gsm(is + 1, d2, m2);
    const uint8_t * qh = base + 16;
    const uint8_t * qs = base + 48 + 32 * q4;
    const int bit_lo = 2 * q4, bit_hi = 2 * q4 + 1;
#pragma unroll
    for (int j = 0; j < 32; j += 2) {
        const uint8_t p0 = qs[j], p1 = qs[j+1], h0 = qh[j], h1 = qh[j+1];
        const float lo0 = fmaf((float)((p0 & 0xF) + (((h0 >> bit_lo) & 1) ? 16 : 0)), d1, -m1);
        const float lo1 = fmaf((float)((p1 & 0xF) + (((h1 >> bit_lo) & 1) ? 16 : 0)), d1, -m1);
        const float hi0 = fmaf((float)((p0 >> 4) + (((h0 >> bit_hi) & 1) ? 16 : 0)), d2, -m2);
        const float hi1 = fmaf((float)((p1 >> 4) + (((h1 >> bit_hi) & 1) ? 16 : 0)), d2, -m2);
        arow[j/2]      = mmb_pack2(lo0, lo1);
        arow[16 + j/2] = mmb_pack2(hi0, hi1);
    }
}

// dequantize one weight row's 64-value quarter of a Q6_K super-block (210 B: ql[128], qh[64],
// scales[16] int8, half d) into 64 bf16.  value = d * scales[..] * (q6 - 32); q6 = 4-bit ql |
// (2-bit qh << 4).  Quarter q4: ip = q4>>1 (half), high = q4&1 (ql nibble), bit sh = high?4:0.
__device__ __forceinline__ void mmb_dq_row_q6k(const uint8_t * base, const int q4, uint32_t * arow) {
    const int ip = q4 >> 1, high = q4 & 1;
    const float d = mmb_h2f(*(const uint16_t *)(base + 208));
    const uint8_t * ql = base + 64 * ip;
    const uint8_t * qh = base + 128 + 32 * ip;
    const int8_t  * sc = (const int8_t *)(base + 192);
    const int sh = high ? 4 : 0;
    const int ib = 8 * ip + (high ? 4 : 0);
#pragma unroll
    for (int il = 0; il < 32; il += 2) {
        float va[2], vb[2];
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const int j = il + t;
            const uint8_t qa = ql[j], qb = ql[32 + j], h = qh[j];
            const int sa = ib + (j >> 4);
            const int a6 = (high ? (qa >> 4) : (qa & 0xF)) | (((h >> sh)       & 3) << 4);
            const int b6 = (high ? (qb >> 4) : (qb & 0xF)) | (((h >> (sh + 2)) & 3) << 4);
            va[t] = d * (float) sc[sa]     * (float)(a6 - 32);
            vb[t] = d * (float) sc[sa + 2] * (float)(b6 - 32);
        }
        arow[il/2]      = mmb_pack2(va[0], va[1]);
        arow[16 + il/2] = mmb_pack2(vb[0], vb[1]);
    }
}

template <typename DRowFn>
__device__ __forceinline__ void mmb_store_tile(const v8f & acc, float * __restrict__ stg, float * __restrict__ D, uint16_t * __restrict__ Dh,
        const bool store_f32, const int M, DRowFn drow, const int n_base, const int m_base, const int lane) {
    const int cm = lane & 15, cn = lane >> 4;
#pragma unroll
    for (int e = 0; e < 8; ++e) { stg[MMB_ACC_M(e, cn) * 16 + cm] = acc[e]; }
    __syncthreads();
    const int n = lane >> 1, half = lane & 1;
    const int dr = drow(n_base + n);
    if (dr >= 0) {
        const float4 v0 = *(const float4 *)(stg + n * 16 + half * 8), v1 = *(const float4 *)(stg + n * 16 + half * 8 + 4);
        const size_t base = (size_t)dr * M + m_base + half * 8;
        const bool full = m_base + 16 <= M;
        if (store_f32) {
            if (full && (M & 3) == 0) { *(float4 *)(D + base) = v0; *(float4 *)(D + base + 4) = v1; }
            else { const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
                   for (int k = 0; k < 8; ++k) { if (m_base + half * 8 + k < M) D[base + k] = vv[k]; } }
        }
        if (Dh) {
            if (full && (M & 7) == 0) { *(uint4 *)(Dh + base) = make_uint4(mmb_pack2(v0.x, v0.y), mmb_pack2(v0.z, v0.w), mmb_pack2(v1.x, v1.y), mmb_pack2(v1.z, v1.w)); }
            else { const float vv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
#pragma unroll
                   for (int k = 0; k < 8; ++k) { if (m_base + half * 8 + k < M) Dh[base + k] = mmb_f2bf(vv[k]); } }
        }
    }
    __syncthreads();
}
template <int BM, int BN, int WTM, int WTN, int WTYPE, bool TAIL, bool DBUF, bool DBUF2, typename XRowFn, typename DRowFn>
__device__ __forceinline__ void mmb_tile_gemm(const uint8_t * __restrict__ Wbase, const size_t wrow_bytes, const int a_rows,
        const uint16_t * __restrict__ Xh, const int K, XRowFn xrow, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32, const int M, DRowFn drow, const int m0,
        const int n_cols, uint16_t * As, uint16_t * Bs) {
    constexpr int WAVES_M = BM / WTM, TM = WTM / 16, TN = WTN / 16;
    constexpr int A_ITEMS = (BM + MMB_NT - 1) / MMB_NT;
    // WTYPE 5 (IQ3_S): when BM < MMB_NT the row-per-thread A mapping leaves warps idle, so the
    // threads sharing a row split its 8 (g,il) groups instead (see mmb_dq_row_iq3s_p).
    constexpr bool SPLIT_A = (WTYPE == 5 && MMB_NT > BM && MMB_NT % BM == 0);
    constexpr int PARTS = SPLIT_A ? (MMB_NT / BM) : 1;
    constexpr int B_ITEMS = (BN * 8) / MMB_NT;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave % WAVES_M, wn = wave / WAVES_M;
    uint4 a0[A_ITEMS], a1[A_ITEMS], a3[A_ITEMS], a4[A_ITEMS], a5[A_ITEMS], a6[A_ITEMS], a7[A_ITEMS], a8[A_ITEMS]; uint32_t a2[A_ITEMS];
    uint4 bst[B_ITEMS];
    int brow[B_ITEMS];
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; brow[i] = xrow(c >> 3); }

    auto load_regs = [&](const int ks) {
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) {
            const int row = tid + i * MMB_NT;
            if (row < BM && row < a_rows) {
                if constexpr (WTYPE == 0) { const uint8_t * p = Wbase + (size_t)row * wrow_bytes + (size_t)ks * 36;
                    a0[i] = *(const uint4 *)(p); a1[i] = *(const uint4 *)(p + 16); a2[i] = *(const uint32_t *)(p + 32); }
                else if constexpr (WTYPE == 1) { const uint8_t * p = Wbase + (size_t)row * wrow_bytes + (size_t)ks * 68;
                    a0[i] = *(const uint4 *)(p); a1[i] = *(const uint4 *)(p + 16); a3[i] = *(const uint4 *)(p + 32); a4[i] = *(const uint4 *)(p + 48); a2[i] = *(const uint32_t *)(p + 64); }
                else if constexpr (WTYPE == 3) { const uint8_t * p = Wbase + (size_t)row * wrow_bytes + (size_t)(ks >> 2) * 144;
                    a0[i] = *(const uint4 *)(p); a1[i] = *(const uint4 *)(p + 16 + (size_t)(ks & 3) * 32); a3[i] = *(const uint4 *)(p + 32 + (size_t)(ks & 3) * 32); }

                else if constexpr (WTYPE == 5) { /* IQ3_S: fields non-contiguous, store_lds loads from Wbase */ }
                else if constexpr (WTYPE >= 6) { /* K-quant / i-quant: store_lds loads from Wbase */ }
                else { const uint4 * p = (const uint4 *)(Wbase + (size_t)row * wrow_bytes + (size_t)ks * 128);
                    a0[i] = p[0]; a1[i] = p[1]; a3[i] = p[2]; a4[i] = p[3]; a5[i] = p[4]; a6[i] = p[5]; a7[i] = p[6]; a8[i] = p[7]; }
            } else { a0[i] = make_uint4(0,0,0,0); a1[i] = make_uint4(0,0,0,0); a3[i] = make_uint4(0,0,0,0); a4[i] = make_uint4(0,0,0,0); a5[i] = a6[i] = a7[i] = a8[i] = make_uint4(0,0,0,0); a2[i] = 0; }
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) {
            const int c = tid + i * MMB_NT; const int off = (c & 7) * 8;
            bst[i] = (brow[i] >= 0) ? *(const uint4 *)(Xh + (size_t)brow[i] * K + ks * MMB_BK + off) : make_uint4(0,0,0,0);
        }
    };
    auto store_lds_a = [&](int ksh, const int abo) {
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) { const int row = SPLIT_A ? (tid % BM) : (tid + i * MMB_NT); if (row < BM) {
            if constexpr (WTYPE == 0) mmb_dq_row36(a0[i], a1[i], a2[i], (uint32_t *)(As + (abo + row) * MMB_LDS_STRIDE));
            else if constexpr (WTYPE == 1) mmb_dq_row68(a0[i], a1[i], a3[i], a4[i], a2[i], (uint32_t *)(As + (abo + row) * MMB_LDS_STRIDE));
            else if constexpr (WTYPE == 3) mmb_dq_row_q4k(a0[i], a1[i], a3[i], ksh, (uint32_t *)(As + (abo + row) * MMB_LDS_STRIDE));

            else if constexpr (SPLIT_A) {
                constexpr int PARTS = MMB_NT / BM;
                mmb_dq_row_iq3s_p<PARTS>(Wbase + (size_t)row * wrow_bytes + (size_t)(ksh >> 2) * 110, ksh & 3,
                                         (uint32_t *)(As + (abo + row) * MMB_LDS_STRIDE), tid / BM);
            }
            else if constexpr (WTYPE == 5) mmb_dq_row_iq3s(Wbase + (size_t)row * wrow_bytes + (size_t)(ksh >> 2) * 110, ksh & 3, (uint32_t *)(As + (abo + row) * MMB_LDS_STRIDE));
            else if constexpr (WTYPE == 6) mmb_dq_row_q5k(Wbase + (size_t)row * wrow_bytes + (size_t)(ksh >> 2) * 176, ksh & 3, (uint32_t *)(As + (abo + row) * MMB_LDS_STRIDE));
            else if constexpr (WTYPE == 7) mmb_dq_row_q6k(Wbase + (size_t)row * wrow_bytes + (size_t)(ksh >> 2) * 210, ksh & 3, (uint32_t *)(As + (abo + row) * MMB_LDS_STRIDE));
            else if constexpr (WTYPE == 8) mmb_dq_row_iq4xs(Wbase + (size_t)row * wrow_bytes + (size_t)(ksh >> 2) * 136, ksh & 3, (uint32_t *)(As + (abo + row) * MMB_LDS_STRIDE));
            else if constexpr (WTYPE == 9) mmb_dq_row_q3k(Wbase + (size_t)row * wrow_bytes + (size_t)(ksh >> 2) * 110, ksh & 3, (uint32_t *)(As + (abo + row) * MMB_LDS_STRIDE));

            else { uint4 * d = (uint4 *)(As + (abo + row) * MMB_LDS_STRIDE); d[0] = a0[i]; d[1] = a1[i]; d[2] = a3[i]; d[3] = a4[i]; d[4] = a5[i]; d[5] = a6[i]; d[6] = a7[i]; d[7] = a8[i]; } } }
    };
    auto store_lds_b = [&](const int bbo) {
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; *(uint4 *)(Bs + (bbo + (c >> 3)) * MMB_LDS_STRIDE + (c & 7) * 8) = bst[i]; }
    };

    v8f acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

    bool jact[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) jact[j] = !TAIL || (wn * WTN + j * 16) < n_cols;   // whole fragments past the valid columns are never stored
    const int nks = K / MMB_BK;
    load_regs(0); store_lds_a(0, 0); store_lds_b(0); __syncthreads();
    for (int ks = 0; ks < nks; ++ks) {
        if (ks + 1 < nks) load_regs(ks + 1);
        // DBUF: the NEXT K-step's weight dequant writes the other A buffer, so it overlaps this
        // step's WMMA instead of serializing behind the A single-buffer barrier hazard (this is the
        // session-21 win; it requires As to be 2x).  DBUF2 additionally double-buffers B to drop the
        // second barrier -- MEASURED REFUTED (GLU 1225 -> 1433 ms): the extra LDS traffic costs more
        // than the removed barrier buys.  Kept guarded so it can be re-tested cheaply.
        if (DBUF && ks + 1 < nks) store_lds_a(ks + 1, ((ks + 1) & 1) * BM);
        if (DBUF2 && ks + 1 < nks) store_lds_b(((ks + 1) & 1) * BN);
        const int ab = DBUF ? ((ks & 1) * BM) : 0;
        const int bb = DBUF2 ? ((ks & 1) * BN) : 0;
#pragma unroll
        for (int kk = 0; kk < MMB_BK; kk += 16) {
            mmb_frag_t a[TM], b[TN]; const int r = lane & 15;
#pragma unroll
            for (int i = 0; i < TM; ++i) { const uint16_t * p = As + (ab + wm * WTM + i * 16 + r) * MMB_LDS_STRIDE + kk;
                a[i] = mmb_ld_frag(p, lane >> 4); }
#pragma unroll
            for (int j = 0; j < TN; ++j) { if constexpr (TAIL) { if (!jact[j]) continue; } const uint16_t * p = Bs + (bb + wn * WTN + j * 16 + r) * MMB_LDS_STRIDE + kk;
                b[j] = mmb_ld_frag(p, lane >> 4); }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) { if constexpr (TAIL) { if (!jact[j]) continue; } acc[i][j] = mmb_wmma_bf16(b[j], a[i], acc[i][j]); }
        }
        __syncthreads();
        if constexpr (!DBUF2) {
            if (!DBUF && ks + 1 < nks) store_lds_a(ks + 1, 0);
            if (ks + 1 < nks) store_lds_b(0);
            __syncthreads();
        }
    }
    // epilogue through LDS (per-wave 1 KB stage in the now-free A/B tile area); tiles are 16 rows, a_rows is a
    // multiple of 32 in this model, so whole tiles are either valid or beyond a_rows
    float * stg = (float *)((BN * MMB_LDS_STRIDE * 2 >= MMB_NT / 32 * 1024) ? Bs : As) + wave * 256;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int ml = wm * WTM + i * 16; const bool ok = ml < a_rows;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            if (ok && jact[j]) mmb_store_tile(acc[i][j], stg, D, Dh, store_f32, M, drow, wn * WTN + j * 16, m0 + ml, lane);
            else { __syncthreads(); __syncthreads(); }
        }
    }
}

template <int BM, int BN, int WTM, int WTN, int WTYPE, bool DBUF = false, bool DBUF2 = false>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_dense_kernel(const uint8_t * __restrict__ W, const uint16_t * __restrict__ Xh, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32, const int M, const int K, const int T) {
    __shared__ __align__(16) uint16_t As[(DBUF ? 2 : 1) * BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[(DBUF2 ? 2 : 1) * BN * MMB_LDS_STRIDE];
    const int m0 = blockIdx.x * BM, t0 = blockIdx.y * BN;
    const size_t wrow_bytes = WTYPE == 2 ? (size_t) K * 2 :
                              WTYPE == 3 ? (size_t)(K / 256) * 144 :
                              WTYPE == 5 ? (size_t)(K / 256) * 110 :
                              WTYPE == 6 ? (size_t)(K / 256) * 176 :
                              WTYPE == 7 ? (size_t)(K / 256) * 210 :
                              WTYPE == 8 ? (size_t)(K / 256) * 136 :
                              WTYPE == 9 ? (size_t)(K / 256) * 110 :
                              (size_t)(K / 32) * (WTYPE == 0 ? 18 : 34);
    mmb_tile_gemm<BM, BN, WTM, WTN, WTYPE, false, DBUF, DBUF2>(W + (size_t)m0 * wrow_bytes, wrow_bytes, M - m0, Xh, K,
        [&](int i) { return (t0 + i < T) ? t0 + i : -1; }, D, Dh, store_f32, M, [&](int i) { return (t0 + i < T) ? t0 + i : -1; }, m0, T - t0, As, Bs);
}

template <int BM, int BN, int WTM, int WTN, int WTYPE, bool DBUF = false, bool DBUF2 = false>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_routed_kernel(const uint8_t * __restrict__ W, const size_t expert_bytes, const uint16_t * __restrict__ Xh, float * __restrict__ D,
        uint16_t * __restrict__ Dh, const bool store_f32,
        const int32_t * __restrict__ ids_src, const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds,
        const uint32_t * __restrict__ desc, const int M, const int K) {
    __shared__ __align__(16) uint16_t As[(DBUF ? 2 : 1) * BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[(DBUF2 ? 2 : 1) * BN * MMB_LDS_STRIDE];
    const uint32_t dsc = desc[blockIdx.y];
    if (dsc == UINT32_MAX) return;   // uniform across the block, before any barrier
    const int e = dsc & 0xffff, jt = dsc >> 16;
    const int r0 = bounds[e] + jt * BN, cnt = bounds[e + 1] - r0;
    const int m0 = blockIdx.x * BM;
    const size_t wrow_bytes = WTYPE == 3 ? (size_t)(K / 256) * 144 : WTYPE == 5 ? (size_t)(K / 256) * 110 : WTYPE == 6 ? (size_t)(K / 256) * 176 : WTYPE == 7 ? (size_t)(K / 256) * 210 : WTYPE == 8 ? (size_t)(K / 256) * 136 : WTYPE == 9 ? (size_t)(K / 256) * 110 : WTYPE == 1 ? (size_t)(K / 32) * 34 : (size_t)(K / 32) * 18;
    mmb_tile_gemm<BM, BN, WTM, WTN, WTYPE, true, DBUF, DBUF2>(W + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, wrow_bytes, M - m0, Xh, K,
        [&](int i) { return (i < cnt) ? ids_src[r0 + i] : -1; }, D, Dh, store_f32, M, [&](int i) { return (i < cnt) ? ids_dst[r0 + i] : -1; }, m0, cnt, As, Bs);
}

template <int BM, int BN, int WTM, int WTN, int WTYPE, bool TAIL, bool DBUF, bool DBUF2, typename XRowFn, typename DRowFn>
__device__ __forceinline__ void mmb_tile_gemm_glu(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu, const size_t wrow_bytes, const int a_rows,
        const uint16_t * __restrict__ Xh, const int K, XRowFn xrow, float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32, const int M, DRowFn drow, const int m0,
        const int n_cols, uint16_t * Ag, uint16_t * Au, uint16_t * Bs) {
    constexpr int WAVES_M = BM / WTM, TM = WTM / 16, TN = WTN / 16;
    constexpr int A_ITEMS = (BM + MMB_NT - 1) / MMB_NT;
    // WTYPE 5 (IQ3_S): when BM < MMB_NT the row-per-thread A mapping leaves warps idle, so the
    // threads sharing a row split its 8 (g,il) groups instead (see mmb_dq_row_iq3s_p).
    constexpr bool SPLIT_A = (WTYPE == 5 && MMB_NT > BM && MMB_NT % BM == 0);
    constexpr int PARTS = SPLIT_A ? (MMB_NT / BM) : 1;
    constexpr int B_ITEMS = (BN * 8) / MMB_NT;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wm = wave % WAVES_M, wn = wave / WAVES_M;
    uint4 g0[A_ITEMS], g1[A_ITEMS], g3[A_ITEMS], g4[A_ITEMS], u0[A_ITEMS], u1[A_ITEMS], u3[A_ITEMS], u4[A_ITEMS]; uint32_t g2[A_ITEMS], u2[A_ITEMS];
    mmb_iq3s_r<PARTS> iqr[2];   // [0] = gate, [1] = up (WTYPE 5 split path only)
    uint4 bst[B_ITEMS];
    int brow[B_ITEMS];
#pragma unroll
    for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; brow[i] = xrow(c >> 3); }
    auto load_regs = [&](const int ks) {
        if constexpr (SPLIT_A) {
            // IQ3_S, split A: preload this thread's block fields for step `ks` into registers, where
            // the loads overlap the previous K-step's WMMA, so store_lds_a becomes pure ALU.  NOTE the
            // row is `tid % BM` (every thread is active) and NOT the A_ITEMS loop's `tid + i*MMB_NT`
            // -- hoisting this out of that loop's `row < BM` guard is load-bearing (put it inside and
            // only threads 0..BM-1 run it, all taking part 0, and the output is garbage).
            const int prow = tid % BM;
            const size_t off = (size_t)prow * wrow_bytes + (size_t)(ks >> 2) * 110;
            iqr[0] = mmb_iq3s_preload<PARTS>(Wg + off, ks & 3, tid / BM);
            iqr[1] = mmb_iq3s_preload<PARTS>(Wu + off, ks & 3, tid / BM);
        }
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) {
            const int row = tid + i * MMB_NT;
            if (row < BM && row < a_rows) {
                if constexpr (WTYPE == 0) {
                    const uint8_t * pg = Wg + (size_t)row * wrow_bytes + (size_t)ks * 36;
                    const uint8_t * pu = Wu + (size_t)row * wrow_bytes + (size_t)ks * 36;
                    g0[i] = *(const uint4 *)(pg); g1[i] = *(const uint4 *)(pg + 16); g2[i] = *(const uint32_t *)(pg + 32);
                    u0[i] = *(const uint4 *)(pu); u1[i] = *(const uint4 *)(pu + 16); u2[i] = *(const uint32_t *)(pu + 32);
                } else if constexpr (WTYPE == 1) {
                    const uint8_t * pg = Wg + (size_t)row * wrow_bytes + (size_t)ks * 68;
                    const uint8_t * pu = Wu + (size_t)row * wrow_bytes + (size_t)ks * 68;
                    g0[i] = *(const uint4 *)(pg); g1[i] = *(const uint4 *)(pg + 16); g3[i] = *(const uint4 *)(pg + 32); g4[i] = *(const uint4 *)(pg + 48); g2[i] = *(const uint32_t *)(pg + 64);
                    u0[i] = *(const uint4 *)(pu); u1[i] = *(const uint4 *)(pu + 16); u3[i] = *(const uint4 *)(pu + 32); u4[i] = *(const uint4 *)(pu + 48); u2[i] = *(const uint32_t *)(pu + 64);
                } else if constexpr (WTYPE == 3) {
                    const uint8_t * pg = Wg + (size_t)row * wrow_bytes + (size_t)(ks >> 2) * 144;
                    const uint8_t * pu = Wu + (size_t)row * wrow_bytes + (size_t)(ks >> 2) * 144;
                    const size_t qo = 16 + (size_t)(ks & 3) * 32;
                    g0[i] = *(const uint4 *)(pg); g1[i] = *(const uint4 *)(pg + qo); g3[i] = *(const uint4 *)(pg + qo + 16);
                    u0[i] = *(const uint4 *)(pu); u1[i] = *(const uint4 *)(pu + qo); u3[i] = *(const uint4 *)(pu + qo + 16);
                }  else if constexpr (WTYPE == 5) { /* IQ3_S: fields non-contiguous, store_lds loads from Wg/Wu */ }
                else if constexpr (WTYPE >= 6) { /* K-quant / i-quant: store_lds loads from Wg/Wu */ }
            } else { g0[i] = g1[i] = u0[i] = u1[i] = g3[i] = u3[i] = g4[i] = u4[i] = make_uint4(0,0,0,0); g2[i] = u2[i] = 0; }
        }
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) {
            const int c = tid + i * MMB_NT; const int off = (c & 7) * 8;
            bst[i] = (brow[i] >= 0) ? *(const uint4 *)(Xh + (size_t)brow[i] * K + ks * MMB_BK + off) : make_uint4(0,0,0,0);
        }
    };
    auto store_lds_a = [&](int ksh, const int abo) {
#pragma unroll
        for (int i = 0; i < A_ITEMS; ++i) { const int row = SPLIT_A ? (tid % BM) : (tid + i * MMB_NT); if (row < BM) {
            if constexpr (WTYPE == 0) {
                mmb_dq_row36(g0[i], g1[i], g2[i], (uint32_t *)(Ag + (abo + row) * MMB_LDS_STRIDE));
                mmb_dq_row36(u0[i], u1[i], u2[i], (uint32_t *)(Au + (abo + row) * MMB_LDS_STRIDE));
            } else if constexpr (WTYPE == 1) {
                mmb_dq_row68(g0[i], g1[i], g3[i], g4[i], g2[i], (uint32_t *)(Ag + (abo + row) * MMB_LDS_STRIDE));
                mmb_dq_row68(u0[i], u1[i], u3[i], u4[i], u2[i], (uint32_t *)(Au + (abo + row) * MMB_LDS_STRIDE));
            } else if constexpr (WTYPE == 3) {
                mmb_dq_row_q4k(g0[i], g1[i], g3[i], ksh, (uint32_t *)(Ag + (abo + row) * MMB_LDS_STRIDE));
                mmb_dq_row_q4k(u0[i], u1[i], u3[i], ksh, (uint32_t *)(Au + (abo + row) * MMB_LDS_STRIDE));
            }  else if constexpr (SPLIT_A) {
                mmb_dq_iq3s_r<PARTS>(iqr[0], (uint32_t *)(Ag + (abo + row) * MMB_LDS_STRIDE), tid / BM);
                mmb_dq_iq3s_r<PARTS>(iqr[1], (uint32_t *)(Au + (abo + row) * MMB_LDS_STRIDE), tid / BM);
            } else if constexpr (WTYPE == 5) {
                mmb_dq_row_iq3s(Wg + (size_t)row * wrow_bytes + (size_t)(ksh >> 2) * 110, ksh & 3, (uint32_t *)(Ag + (abo + row) * MMB_LDS_STRIDE));
                mmb_dq_row_iq3s(Wu + (size_t)row * wrow_bytes + (size_t)(ksh >> 2) * 110, ksh & 3, (uint32_t *)(Au + (abo + row) * MMB_LDS_STRIDE));
            } else if constexpr (WTYPE == 6) {
                mmb_dq_row_q5k(Wg + (size_t)row * wrow_bytes + (size_t)(ksh >> 2) * 176, ksh & 3, (uint32_t *)(Ag + (abo + row) * MMB_LDS_STRIDE));
                mmb_dq_row_q5k(Wu + (size_t)row * wrow_bytes + (size_t)(ksh >> 2) * 176, ksh & 3, (uint32_t *)(Au + (abo + row) * MMB_LDS_STRIDE));
            } else if constexpr (WTYPE == 7) {
                mmb_dq_row_q6k(Wg + (size_t)row * wrow_bytes + (size_t)(ksh >> 2) * 210, ksh & 3, (uint32_t *)(Ag + (abo + row) * MMB_LDS_STRIDE));
                mmb_dq_row_q6k(Wu + (size_t)row * wrow_bytes + (size_t)(ksh >> 2) * 210, ksh & 3, (uint32_t *)(Au + (abo + row) * MMB_LDS_STRIDE));
            } else if constexpr (WTYPE == 8) {
                mmb_dq_row_iq4xs(Wg + (size_t)row * wrow_bytes + (size_t)(ksh >> 2) * 136, ksh & 3, (uint32_t *)(Ag + (abo + row) * MMB_LDS_STRIDE));
                mmb_dq_row_iq4xs(Wu + (size_t)row * wrow_bytes + (size_t)(ksh >> 2) * 136, ksh & 3, (uint32_t *)(Au + (abo + row) * MMB_LDS_STRIDE));
            } else if constexpr (WTYPE == 9) {
                mmb_dq_row_q3k(Wg + (size_t)row * wrow_bytes + (size_t)(ksh >> 2) * 110, ksh & 3, (uint32_t *)(Ag + (abo + row) * MMB_LDS_STRIDE));
                mmb_dq_row_q3k(Wu + (size_t)row * wrow_bytes + (size_t)(ksh >> 2) * 110, ksh & 3, (uint32_t *)(Au + (abo + row) * MMB_LDS_STRIDE));
            }          } }
    };
    auto store_lds_b = [&](const int bbo) {
#pragma unroll
        for (int i = 0; i < B_ITEMS; ++i) { const int c = tid + i * MMB_NT; *(uint4 *)(Bs + (bbo + (c >> 3)) * MMB_LDS_STRIDE + (c & 7) * 8) = bst[i]; }
    };
    v8f accg[TM][TN], accu[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) { accg[i][j][e] = 0.f; accu[i][j][e] = 0.f; }
    bool jact[TN];
#pragma unroll
    for (int j = 0; j < TN; ++j) jact[j] = !TAIL || (wn * WTN + j * 16) < n_cols;
    const int nks = K / MMB_BK;
    load_regs(0); store_lds_a(0, 0); store_lds_b(0); __syncthreads();
    for (int ks = 0; ks < nks; ++ks) {
        if (ks + 1 < nks) load_regs(ks + 1);
        // DBUF: the NEXT K-step's gate+up dequant writes the other A buffers, overlapping this step's
        // WMMA.  Only enabled for WTYPE 5 (IQ3_S): it is a -6.7 % win on the Flash-Next IQ3_S GLU but
        // a +5.8 % loss on the 35B Q4_K GLU.  DBUF2 (double-buffer B, one sync) is MEASURED REFUTED.
        if (DBUF && ks + 1 < nks) store_lds_a(ks + 1, ((ks + 1) & 1) * BM);
        if (DBUF2 && ks + 1 < nks) store_lds_b(((ks + 1) & 1) * BN);
        const int ab = DBUF ? ((ks & 1) * BM) : 0;
        const int bb = DBUF2 ? ((ks & 1) * BN) : 0;
#pragma unroll
        for (int kk = 0; kk < MMB_BK; kk += 16) {
            mmb_frag_t ag[TM], au[TM], b[TN]; const int r = lane & 15;
#pragma unroll
            for (int i = 0; i < TM; ++i) { const int off = (ab + wm * WTM + i * 16 + r) * MMB_LDS_STRIDE + kk;
                ag[i] = mmb_ld_frag(Ag + off, lane >> 4);
                au[i] = mmb_ld_frag(Au + off, lane >> 4); }
#pragma unroll
            for (int j = 0; j < TN; ++j) { if constexpr (TAIL) { if (!jact[j]) continue; } const uint16_t * p = Bs + (bb + wn * WTN + j * 16 + r) * MMB_LDS_STRIDE + kk;
                b[j] = mmb_ld_frag(p, lane >> 4); }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    if constexpr (TAIL) { if (!jact[j]) continue; }
                    accg[i][j] = mmb_wmma_bf16(b[j], ag[i], accg[i][j]);
                    accu[i][j] = mmb_wmma_bf16(b[j], au[i], accu[i][j]);
                }
        }
        __syncthreads();
        if constexpr (!DBUF2) {
            if (!DBUF && ks + 1 < nks) store_lds_a(ks + 1, 0);
            if (ks + 1 < nks) store_lds_b(0);
            __syncthreads();
        }
    }
    float * stg = (float *)((BN * MMB_LDS_STRIDE * 2 >= MMB_NT / 32 * 1024) ? Bs : Ag) + wave * 256;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int ml = wm * WTM + i * 16; const bool ok = ml < a_rows;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            if (ok && jact[j]) {
                v8f v;
#pragma unroll
                for (int e = 0; e < 8; ++e) { v[e] = ggml_cuda_op_silu_single(accg[i][j][e]) * accu[i][j][e]; }
                mmb_store_tile(v, stg, D, Dh, store_f32, M, drow, wn * WTN + j * 16, m0 + ml, lane);
            } else { __syncthreads(); __syncthreads(); }
        }
    }
}

template <int BM, int BN, int WTM, int WTN, int WTYPE, bool DBUF = false, bool DBUF2 = false>
__global__ void __launch_bounds__(MMB_NT, 2)
mmb_routed_glu_kernel(const uint8_t * __restrict__ Wg, const uint8_t * __restrict__ Wu, const size_t expert_bytes, const uint16_t * __restrict__ Xh,
        float * __restrict__ D, uint16_t * __restrict__ Dh, const bool store_f32,
        const int32_t * __restrict__ ids_src, const int32_t * __restrict__ ids_dst, const int32_t * __restrict__ bounds,
        const uint32_t * __restrict__ desc, const int M, const int K) {
    __shared__ __align__(16) uint16_t Ag[(DBUF ? 2 : 1) * BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Au[(DBUF ? 2 : 1) * BM * MMB_LDS_STRIDE];
    __shared__ __align__(16) uint16_t Bs[(DBUF2 ? 2 : 1) * BN * MMB_LDS_STRIDE];
    const uint32_t dsc = desc[blockIdx.y];
    if (dsc == UINT32_MAX) return;
    const int e = dsc & 0xffff, jt = dsc >> 16;
    const int r0 = bounds[e] + jt * BN, cnt = bounds[e + 1] - r0;
    const int m0 = blockIdx.x * BM;
    const size_t wrow_bytes = WTYPE == 3 ? (size_t)(K / 256) * 144 : WTYPE == 5 ? (size_t)(K / 256) * 110 : WTYPE == 6 ? (size_t)(K / 256) * 176 : WTYPE == 7 ? (size_t)(K / 256) * 210 : WTYPE == 8 ? (size_t)(K / 256) * 136 : WTYPE == 9 ? (size_t)(K / 256) * 110 : WTYPE == 1 ? (size_t)(K / 32) * 34 : (size_t)(K / 32) * 18;
    mmb_tile_gemm_glu<BM, BN, WTM, WTN, WTYPE, true, DBUF, DBUF2>(Wg + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, Wu + (size_t)e * expert_bytes + (size_t)m0 * wrow_bytes, wrow_bytes, M - m0, Xh, K,
        [&](int i) { return (i < cnt) ? ids_src[r0 + i] : -1; }, D, Dh, store_f32, M, [&](int i) { return (i < cnt) ? ids_dst[r0 + i] : -1; }, m0, cnt, Ag, Au, Bs);
}

__global__ void mmb_build_desc2(const int32_t * __restrict__ bounds, uint32_t * __restrict__ desc_big, uint32_t * __restrict__ desc_small,
        const int E, const int nbig_max, const int nsmall_max, const int BN_BIG, const int BN_SMALL, const int thresh) {
    __shared__ int sb[1024], ss[1024];
    const int e = threadIdx.x;
    for (int i = e; i < nbig_max;   i += blockDim.x) desc_big[i]   = UINT32_MAX;
    for (int i = e; i < nsmall_max; i += blockDim.x) desc_small[i] = UINT32_MAX;
    int cnt = (e < E) ? bounds[e + 1] - bounds[e] : 0;
    const bool big = cnt >= thresh;
    const int tb = big ? (cnt + BN_BIG - 1) / BN_BIG : 0;
    const int ts = big ? 0 : (cnt + BN_SMALL - 1) / BN_SMALL;
    sb[e] = tb; ss[e] = ts;
    __syncthreads();
    for (int off = 1; off < 1024; off <<= 1) {
        const int vb = (e >= off) ? sb[e - off] : 0, vs = (e >= off) ? ss[e - off] : 0;
        __syncthreads();
        sb[e] += vb; ss[e] += vs;
        __syncthreads();
    }
    const int bb = sb[e] - tb, bs = ss[e] - ts;
    for (int jt = 0; jt < tb; ++jt) { const int idx = bb + jt; if (idx < nbig_max)   desc_big[idx]   = (uint32_t)e | ((uint32_t)jt << 16); }
    for (int jt = 0; jt < ts; ++jt) { const int idx = bs + jt; if (idx < nsmall_max) desc_small[idx] = (uint32_t)e | ((uint32_t)jt << 16); }
}

} // namespace

static int mmb_weight_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ4_NL: return 0;
        case GGML_TYPE_Q8_0:   return 1;
        case GGML_TYPE_BF16:   return 2;
        case GGML_TYPE_Q4_K:   return 3;
        case GGML_TYPE_IQ3_S:  return 5;
        case GGML_TYPE_Q5_K:   return 6;
        case GGML_TYPE_Q6_K:   return 7;
        case GGML_TYPE_IQ4_XS: return 8;
        case GGML_TYPE_Q3_K:   return 9;
        default: return -1;
    }
}

static bool mmb_weight_supported(int device, const ggml_tensor * w) {
    const auto & dev = ggml_cuda_info().devices[device];
    return GGML_CUDA_CC_IS_RDNA3_0(dev.cc) && dev.warp_size == 32 && mmb_weight_type(w->type) >= 0 &&
        ggml_is_contiguous(w) && w->ne[0] > 0 && w->ne[0] <= INT32_MAX &&
        w->ne[0] % 256 == 0 && w->ne[1] > 0 && w->ne[1] <= INT32_MAX && w->ne[1] % 128 == 0;
}

bool ggml_cuda_mmb_supported_mm(int device, const ggml_tensor * w, const ggml_tensor * x, const ggml_tensor * dst) {
    return mmb_weight_supported(device, w) && w->ne[2] == 1 && w->ne[3] == 1 &&
        x->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && ggml_is_contiguous(dst) &&
        x->ne[0] == w->ne[0] && dst->ne[0] == w->ne[1] && ggml_nrows(x) == ggml_nrows(dst) &&
        ggml_nrows(x) >= 512 && ggml_nrows(x) <= 65536;
}

bool ggml_cuda_mmb_supported_mmid(int device, const ggml_tensor * w, const ggml_tensor * x, const ggml_tensor * ids, const ggml_tensor * dst) {
    if (!mmb_weight_supported(device, w) || w->type == GGML_TYPE_BF16 || w->ne[3] != 1 || w->ne[2] < 1 || w->ne[2] > 1024 ||
        x->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
        !ggml_is_contiguous(x) || !ggml_is_contiguous(dst) || ids->nb[0] != sizeof(int32_t) ||
        ids->nb[1] % sizeof(int32_t) != 0 || ids->nb[1] / sizeof(int32_t) > INT32_MAX) {
        return false;
    }
    const int64_t used = ids->ne[0], tokens = ids->ne[1];
    return used > 0 && used <= 64 && used <= w->ne[2] && tokens >= 512 && tokens <= INT32_MAX / used &&
        tokens <= (int64_t) ggml_cuda_info().devices[device].smpbo / sizeof(uint32_t) &&
        used * tokens < 65536 && ids->ne[2] == 1 && ids->ne[3] == 1 &&
        x->ne[0] == w->ne[0] && (x->ne[1] == 1 || x->ne[1] == used) && x->ne[2] == tokens && x->ne[3] == 1 &&
        dst->ne[0] == w->ne[1] && dst->ne[1] == used && dst->ne[2] == tokens && dst->ne[3] == 1;
}

bool ggml_cuda_mmb_supported_glu(int device, const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * x, const ggml_tensor * ids, const ggml_tensor * dst) {
    return gate && up && x && ids && dst && gate->type == up->type && ggml_are_same_shape(gate, up) &&
        dst->op == GGML_OP_GLU && ggml_get_glu_op(dst) == GGML_GLU_OP_SWIGLU && ggml_get_op_params_i32(dst, 1) == 0 &&
        dst->src[0] && dst->src[1] && dst->src[0]->op == GGML_OP_MUL_MAT_ID && dst->src[1]->op == GGML_OP_MUL_MAT_ID &&
        dst->src[0]->src[0] == gate && dst->src[1]->src[0] == up &&
        dst->src[0]->src[1] == x && dst->src[1]->src[1] == x && dst->src[0]->src[2] == ids && dst->src[1]->src[2] == ids &&
        ggml_cuda_mmb_supported_mmid(device, gate, x, ids, dst) && ggml_cuda_mmb_supported_mmid(device, up, x, ids, dst);
}

template<int WT>
static void mmb_launch(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x, const ggml_tensor * ids, ggml_tensor * dst, const ggml_tensor * up) {
    const int k = w->ne[0], m = w->ne[1];
    const int tokens = ids ? ids->ne[1] : ggml_nrows(x);
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<uint16_t> converted(ctx.pool(), ggml_nelements(x));
    mmb_cvt_f32_bf16<<<(ggml_nelements(x) + 2047) / 2048, 256, 0, stream>>>((const float *) x->data, converted.get(), ggml_nelements(x));
    const uint8_t * weights = (const uint8_t *) w->data;
    float * output = (float *) dst->data;
    if (!ids) {
        // The larger tile spills on gfx1100 for the other quantizations.
        if constexpr (WT == 3) {
            if (m >= 6144 && k >= 2560) {
                mmb_dense_kernel<128, 256, 64, 64, WT><<<dim3(m / 128, (tokens + 255) / 256), MMB_NT, 0, stream>>>(weights, converted.get(), output, nullptr, true, m, k, tokens);
                CUDA_CHECK(cudaGetLastError());
                return;
            }
        }
        mmb_dense_kernel<128, 128, 32, 64, WT><<<dim3(m / 128, (tokens + 127) / 128), MMB_NT, 0, stream>>>(weights, converted.get(), output, nullptr, true, m, k, tokens);
    } else {
        const int experts = w->ne[2], used = ids->ne[0], rows = used * tokens;
        CUDA_CHECK(cudaMemsetAsync(output, 0, ggml_nbytes(dst), stream));
        ggml_cuda_pool_alloc<int32_t> source(ctx.pool(), rows), target(ctx.pool(), rows), bounds(ctx.pool(), experts + 1);
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, source.get(), target.get(), bounds.get(),
            experts, tokens, used, x->ne[1], ids->nb[1] / sizeof(int32_t), x->nb[2] / x->nb[1], false, stream);
        constexpr int big_n = 128, small_n = 32, threshold = 32;
        const int nbig = rows / big_n + experts + 1, nsmall = experts + 1;
        ggml_cuda_pool_alloc<uint32_t> big(ctx.pool(), nbig), small(ctx.pool(), nsmall);
        mmb_build_desc2<<<1, 1024, 0, stream>>>(bounds.get(), big.get(), small.get(), experts, nbig, nsmall, big_n, small_n, threshold);
        if (up) {
            const uint8_t * up_weights = (const uint8_t *) up->data;
            mmb_routed_glu_kernel<64, 128, 16, 64, WT><<<dim3(m / 64, nbig), MMB_NT, 0, stream>>>(weights, up_weights, w->nb[2], converted.get(), output, nullptr, true, source.get(), target.get(), bounds.get(), big.get(), m, k);
            mmb_routed_glu_kernel<64, 32, 16, 16, WT><<<dim3(m / 64, nsmall), MMB_NT, 0, stream>>>(weights, up_weights, w->nb[2], converted.get(), output, nullptr, true, source.get(), target.get(), bounds.get(), small.get(), m, k);
        } else {
            mmb_routed_kernel<128, 128, 32, 64, WT><<<dim3(m / 128, nbig), MMB_NT, 0, stream>>>(weights, w->nb[2], converted.get(), output, nullptr, true, source.get(), target.get(), bounds.get(), big.get(), m, k);
            mmb_routed_kernel<128, 32, 32, 16, WT><<<dim3(m / 128, nsmall), MMB_NT, 0, stream>>>(weights, w->nb[2], converted.get(), output, nullptr, true, source.get(), target.get(), bounds.get(), small.get(), m, k);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}

static void mmb_dispatch(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x, const ggml_tensor * ids, ggml_tensor * dst, const ggml_tensor * up = nullptr) {
#define MMB_CASE(TYPE, WT) case TYPE: mmb_launch<WT>(ctx, w, x, ids, dst, up); break
    switch (w->type) {
        MMB_CASE(GGML_TYPE_IQ4_NL, 0);
        MMB_CASE(GGML_TYPE_Q8_0, 1);
        MMB_CASE(GGML_TYPE_BF16, 2);
        MMB_CASE(GGML_TYPE_Q4_K, 3);
        MMB_CASE(GGML_TYPE_IQ3_S, 5);
        MMB_CASE(GGML_TYPE_Q5_K, 6);
        MMB_CASE(GGML_TYPE_Q6_K, 7);
        MMB_CASE(GGML_TYPE_IQ4_XS, 8);
        MMB_CASE(GGML_TYPE_Q3_K, 9);
        default: GGML_ABORT("unsupported MMB weight type");
    }
#undef MMB_CASE
}

void ggml_cuda_mul_mat_mmb(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x, ggml_tensor * dst) {
    mmb_dispatch(ctx, w, x, nullptr, dst);
}
void ggml_cuda_mul_mat_id_mmb(ggml_backend_cuda_context & ctx, const ggml_tensor * w, const ggml_tensor * x, const ggml_tensor * ids, ggml_tensor * dst) {
    mmb_dispatch(ctx, w, x, ids, dst);
}
void ggml_cuda_mul_mat_id_mmb_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up, const ggml_tensor * x, const ggml_tensor * ids, ggml_tensor * dst) {
    mmb_dispatch(ctx, gate, x, ids, dst, up);
}
#endif
