#include "include/iqk_gemm.hpp"
#include <cstring>

#ifdef IQK_IMPLEMENT

#include "ggml-impl.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#ifdef __x86_64__ // ============================= x86_64

namespace {

#ifdef HAVE_FANCY_SIMD // ================================ AVX

struct IQXKScales {
    IQXKScales(uint8_t shift, int8_t min_val) : eshift(_mm256_set1_epi16(shift)), min(_mm256_set1_epi16(min_val)) {}
    template <typename Q8>
    inline void process(int i, float d, uint16_t extra, __m128i scales8, const Q8& q8, __m256 * accm, __m512i * scales) const {
        auto scales16 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales8, scale_shuffle));
        scales16 = _mm256_mullo_epi16(scales16, _mm256_mask_add_epi16(min, extra, min, eshift));
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            const __m256i prod  = _mm256_madd_epi16(scales16, q8.load_bsums(iy, i));
            accm[iy] = _mm256_fmadd_ps(_mm256_set1_ps(d * q8.scale(iy, i)), _mm256_cvtepi32_ps(prod), accm[iy]);
        }
        scales16 = MM256_SET_M128I(scales8, scales8);
        scales[0] = _mm512_cvtepi8_epi16(_mm256_shuffle_epi8(scales16, shuffle1));
        scales[1] = _mm512_cvtepi8_epi16(_mm256_shuffle_epi8(scales16, shuffle2));
    }
    const __m256i eshift;
    const __m256i min;
    const __m128i scale_shuffle = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);
    const __m128i emask         = _mm_set_epi32(0x80804040, 0x20201010, 0x08080404, 0x02020101);
    const __m128i eshuffle      = _mm_set_epi32(0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200);
    const __m256i shuffle1      = _mm256_set_epi64x(0x0b0b0b0b09090909, 0x0303030301010101, 0x0a0a0a0a08080808, 0x0202020200000000);
    const __m256i shuffle2      = _mm256_set_epi64x(0x0f0f0f0f0d0d0d0d, 0x0707070705050505, 0x0e0e0e0e0c0c0c0c, 0x0606060604040404);
};

struct IQXKScales2 {
    IQXKScales2(uint8_t shift, int8_t min_val) : eshift(_mm256_set1_epi16(shift)), min(_mm256_set1_epi16(min_val)) {}
    template <typename Q8>
    inline void process(int i, float d, uint16_t extra, __m128i scales8, const Q8& q8, __m256 * accm, __m512i * scales) const {
        process(i, d, extra, _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales8, scale_shuffle)), q8, accm, scales);
    }
    template <typename Q8>
    inline void process(int i, float d, uint16_t extra, __m256i scales16, const Q8& q8, __m256 * accm, __m512i * scales) const {
        auto scales_s = _mm256_mullo_epi16(scales16, _mm256_mask_add_epi16(min, extra, min, eshift));
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            const __m256i prod  = _mm256_madd_epi16(scales_s, q8.load_bsums(iy, i));
            accm[iy] = _mm256_fmadd_ps(_mm256_set1_ps(d * q8.scale(iy, i)), _mm256_cvtepi32_ps(prod), accm[iy]);
        }
        auto aux_1 = MM256_SET_M128I(_mm256_castsi256_si128(scales16), _mm256_castsi256_si128(scales16));
        auto aux_2 = MM256_SET_M128I(_mm256_extracti128_si256(scales16, 1), _mm256_extracti128_si256(scales16, 1));
        auto scales16_1 = _mm512_inserti32x8(_mm512_castsi256_si512(aux_1), aux_1, 1);
        auto scales16_2 = _mm512_inserti32x8(_mm512_castsi256_si512(aux_2), aux_2, 1);
        scales[0] = _mm512_shuffle_epi8(scales16_1, shuffles[0]);
        scales[1] = _mm512_shuffle_epi8(scales16_1, shuffles[1]);
        scales[2] = _mm512_shuffle_epi8(scales16_2, shuffles[0]);
        scales[3] = _mm512_shuffle_epi8(scales16_2, shuffles[1]);
    }
    const __m256i eshift;
    const __m256i min;
    const __m128i scale_shuffle = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);
    const __m128i emask         = _mm_set_epi32(0x80804040, 0x20201010, 0x08080404, 0x02020101);
    const __m128i eshuffle      = _mm_set_epi32(0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200);
    const __m512i shuffles[2] = {
        _mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_setzero_si512(),
                            _mm_set1_epi16(0x0100), 0), _mm_set1_epi16(0x0302), 1), _mm_set1_epi16(0x0504), 2), _mm_set1_epi16(0x0706), 3),
        _mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_setzero_si512(),
                            _mm_set1_epi16(0x0908), 0), _mm_set1_epi16(0x0b0a), 1), _mm_set1_epi16(0x0d0c), 2), _mm_set1_epi16(0x0f0e), 3)
    };
};

struct DequantizerIQ3KS final : public BaseDequantizer<block_iq3_ks, true, true> {
    DequantizerIQ3KS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_values()) {}
    template <typename Q8>
    inline void compute_block(int i, const Q8& q8, __m512 * acc) {
        uint32_t aux32; std::memcpy(&aux32, x[i].scales, 4);
        auto scl = _mm_srlv_epi32(_mm_set1_epi32(aux32), _mm_set_epi32(0, 0, 4, 0));
        auto scales128 = _mm_cvtepu8_epi16(_mm_and_si128(scl, _mm_set1_epi8(0xf)));
        scales128 = _mm_mask_add_epi16(scales128, __mmask8(x[i].extra & 0xff), scales128, _mm_set1_epi16(16));
        scales128 = _mm_sub_epi16(scales128, _mm_set1_epi16(16));
        auto shifts = _mm_mask_add_epi16(m64, __mmask8(x[i].extra >> 8), m64, _mm_set1_epi16(4));
        auto mins128 = _mm_mullo_epi16(scales128, shifts);
        auto mins = MM256_SET_M128I(_mm_shuffle_epi8(mins128, s8k.shuffles[1]), _mm_shuffle_epi8(mins128, s8k.shuffles[0]));
        auto scales256 = MM256_SET_M128I(scales128, scales128);
        auto all_scales = _mm512_inserti32x8(_mm512_castsi256_si512(scales256), scales256, 1);
        __m512i scales[4];
        for (int k = 0; k < 4; ++k) scales[k] = _mm512_shuffle_epi8(all_scales, shuffles[k]);
        prepare(x[i].qs, x[i].qh);
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            auto q8s = q8.load_bsums(iy, i);
            auto prod = _mm256_madd_epi16(mins, q8s);
            auto sumi = _mm512_inserti32x8(_mm512_setzero_si512(), prod, 0);
            for (int k = 0; k < 4; ++k) {
                auto p = _mm512_maddubs_epi16(bits.values[k], q8.load_quants64(iy, i, k));
                sumi = _mm512_dpwssd_epi32(sumi, p, scales[k]);
            }
            acc[iy] = _mm512_fmadd_ps(_mm512_set1_ps(d*q8.scale(iy, i)), _mm512_cvtepi32_ps(sumi), acc[iy]);
        }
    }
    inline void prepare(const uint8_t * q2, const uint8_t * qh) {
        bits.prepare(q2);
        auto h256 = _mm256_loadu_si256((const __m256i *)qh);
        auto hbits = _mm512_inserti32x8(_mm512_castsi256_si512(h256), _mm256_srli_epi16(h256, 1), 1);
        bits.values[0] = _mm512_or_si512(bits.values[0], _mm512_and_si512(_mm512_slli_epi16(hbits, 2), hmask));
        bits.values[1] = _mm512_or_si512(bits.values[1], _mm512_and_si512(hbits, hmask));
        bits.values[2] = _mm512_or_si512(bits.values[2], _mm512_and_si512(_mm512_srli_epi16(hbits, 2), hmask));
        bits.values[3] = _mm512_or_si512(bits.values[3], _mm512_and_si512(_mm512_srli_epi16(hbits, 4), hmask));
        bits.values[0] = _mm512_shuffle_epi8(values, bits.values[0]);
        bits.values[1] = _mm512_shuffle_epi8(values, bits.values[1]);
        bits.values[2] = _mm512_shuffle_epi8(values, bits.values[2]);
        bits.values[3] = _mm512_shuffle_epi8(values, bits.values[3]);
    }
    static inline __m512i load_values() {
        static const uint8_t kvalues_iq3nl[16] = {1, 24, 41, 54, 65, 77, 92, 111, 5, 28, 45, 58, 69, 81, 96, 115};
        auto val128 = _mm_loadu_si128((const __m128i *)kvalues_iq3nl);
        auto val256 = MM256_SET_M128I(val128, val128);
        return _mm512_inserti32x8(_mm512_castsi256_si512(val256), val256, 1);
    }

    Q2Bits bits;
    Scales8KBase s8k;

    const __m128i m64 = _mm_set1_epi16(-64);
    const __m512i values;
    const __m512i hmask = _mm512_set1_epi8(4);
    const __m512i shuffles[4] = {
        _mm512_inserti32x8(_mm512_set1_epi16(0x0100), _mm256_set1_epi16(0x0302), 1),
        _mm512_inserti32x8(_mm512_set1_epi16(0x0504), _mm256_set1_epi16(0x0706), 1),
        _mm512_inserti32x8(_mm512_set1_epi16(0x0908), _mm256_set1_epi16(0x0b0a), 1),
        _mm512_inserti32x8(_mm512_set1_epi16(0x0d0c), _mm256_set1_epi16(0x0f0e), 1),
    };
};

#else // ================================= NO AVX

inline void prepare_scales_16(const __m256i& all_scales, __m256i * scales) {
    const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
    const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
    scales[0] = MM256_SET_M128I(l_scales, l_scales);
    scales[1] = MM256_SET_M128I(h_scales, h_scales);
}

struct IQXKScales {
    IQXKScales(int8_t shift, int8_t min_val) : min(_mm256_set1_epi16(min_val)), eshift(_mm_set1_epi8(shift)) {}
    template <typename Q8>
    inline void process(int i, float d, uint16_t extra, __m128i scales8, const Q8& q8, __m256 * accm, __m256i * scales) const {
        auto scales16 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales8, hshuff));
        process(i, d, extra, scales16, q8, accm, scales);
    }
    template <typename Q8>
    inline void process(int i, float d, uint16_t extra, __m256i scales16, const Q8& q8, __m256 * accm, __m256i * scales) const {
        auto extra128 = _mm_set1_epi16(extra);
        extra128 = _mm_cmpeq_epi8(_mm_and_si128(extra128, emask), emask);
        extra128 = _mm_and_si128(extra128, eshift);
        extra128 = _mm_shuffle_epi8(extra128, eshuffle);
        auto scales_s = _mm256_mullo_epi16(scales16, _mm256_add_epi16(min, _mm256_cvtepi8_epi16(extra128)));
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            const __m256i prod  = _mm256_madd_epi16(scales_s, q8.load_bsums(iy, i));
            accm[iy] = _mm256_fmadd_ps(_mm256_set1_ps(d * q8.scale(iy, i)), _mm256_cvtepi32_ps(prod), accm[iy]);
        }
        prepare_scales_16(scales16, scales);
    }

    const __m256i min;
    const __m128i eshift;
    const __m128i hshuff   = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);
    const __m128i emask    = _mm_set_epi32(0x80804040, 0x20201010, 0x08080404, 0x02020101);
    const __m128i eshuffle = _mm_set_epi32(0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200);
};
struct DequantizerIQ3KS final : public BaseDequantizer<block_iq3_ks, true, true> {
    DequantizerIQ3KS(const void * vx, size_t bx) : BaseDequantizer(vx, bx), values(load_values()) {}
    template <typename Q8>
    inline __m256i new_block(int i, [[maybe_unused]] const Q8& q8, [[maybe_unused]] __m256 * accd) {
        uint32_t aux32; std::memcpy(&aux32, x[i].scales, 4);
        auto scl = _mm_cvtepi8_epi16(_mm_and_si128(_mm_srlv_epi32(_mm_set1_epi32(aux32), _mm_set_epi32(0, 0, 4, 0)), _mm_set1_epi8(0xf)));
        auto sch = _mm_cmpeq_epi16(_mm_and_si128(_mm_set1_epi16(x[i].extra), mask), mask);
        auto scales128 = _mm_add_epi16(scl, _mm_and_si128(sch, _mm_set1_epi16(16)));
        scales128 = _mm_sub_epi16(scales128, _mm_set1_epi16(16));
        return MM256_SET_M128I(scales128, scales128);
    }
    inline void prepare(int i, int j) {
        uint8_t extra = x[i].extra >> (8 + 4*j);
        hbits = j == 0 ? _mm256_loadu_si256((const __m256i *)x[i].qh) : _mm256_srli_epi16(hbits, 4);
        bits.prepare(x[i].qs, j);
        bits.values[0] = _mm256_add_epi8(_mm256_set1_epi8((extra << 3) & 8), _mm256_or_si256(bits.values[0], _mm256_and_si256(_mm256_slli_epi16(hbits, 2), mh)));
        bits.values[1] = _mm256_add_epi8(_mm256_set1_epi8((extra << 2) & 8), _mm256_or_si256(bits.values[1], _mm256_and_si256(_mm256_slli_epi16(hbits, 1), mh)));
        bits.values[2] = _mm256_add_epi8(_mm256_set1_epi8((extra << 1) & 8), _mm256_or_si256(bits.values[2], _mm256_and_si256(hbits, mh)));
        bits.values[3] = _mm256_add_epi8(_mm256_set1_epi8((extra << 0) & 8), _mm256_or_si256(bits.values[3], _mm256_and_si256(_mm256_srli_epi16(hbits, 1), mh)));
        for (int k = 0; k < 4; ++k) bits.values[k] = _mm256_shuffle_epi8(values, bits.values[k]);
    }
    inline __m256i load_values() {
        auto v = _mm_loadu_si128((const __m128i *)iq3nl_values);
        return MM256_SET_M128I(v, v);
    }


    Q2Bits bits;
    __m256i hbits;
    const __m256i values;
    const __m256i mh   = _mm256_set1_epi8(4);
    const __m128i mask = _mm_setr_epi16(0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80);
};
#endif



template <int nrc_y>
//IQK_ALWAYS_INLINE void iq234_k_accum_mins(int ibl, __m256i i8scales1, __m256i i8scales2, const Q8<nrc_y, block_q8_K>& q8, __m256i shuff,
inline void iq234_k_accum_mins(int ibl, __m256i i8scales1, __m256i i8scales2, const Q8<nrc_y, block_q8_K>& q8, __m256i shuff,
        __m256i * isum, int16_t min) {
    auto t1 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(i8scales1, 0)), shuff); // blocks  0,  1,  2,  3 for each row
    auto t2 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(i8scales1, 1)), shuff); // blocks  4,  5,  6,  7 for each row
    auto t3 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(i8scales2, 0)), shuff); // blocks  8,  9, 10, 11 for each row
    auto t4 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(i8scales2, 1)), shuff); // blocks 12, 13, 14, 15 for each row
    if constexpr (nrc_y == 1) {
        auto s1 = MM256_SET_M128I(_mm256_extracti128_si256(t3, 0), _mm256_extracti128_si256(t1, 0)); // blocks 0, 1,  8, 9
        auto s2 = MM256_SET_M128I(_mm256_extracti128_si256(t3, 1), _mm256_extracti128_si256(t1, 1)); // blocks 2, 3, 10, 11
        auto s3 = MM256_SET_M128I(_mm256_extracti128_si256(t4, 0), _mm256_extracti128_si256(t2, 0)); // blocks 4, 5, 12, 13
        auto s4 = MM256_SET_M128I(_mm256_extracti128_si256(t4, 1), _mm256_extracti128_si256(t2, 1)); // blocks 6, 7, 14, 15
        auto sumi = _mm256_setzero_si256();
        auto bsums = q8.load_bsums(0, ibl);
#ifdef HAVE_FANCY_SIMD
        sumi = _mm256_dpwssd_epi32(sumi, s1, _mm256_shuffle_epi32(bsums, 0x00));
        sumi = _mm256_dpwssd_epi32(sumi, s2, _mm256_shuffle_epi32(bsums, 0x55));
        sumi = _mm256_dpwssd_epi32(sumi, s3, _mm256_shuffle_epi32(bsums, 0xaa));
        sumi = _mm256_dpwssd_epi32(sumi, s4, _mm256_shuffle_epi32(bsums, 0xff));
#else
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s1, _mm256_shuffle_epi32(bsums, 0x00)));
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s2, _mm256_shuffle_epi32(bsums, 0x55)));
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s3, _mm256_shuffle_epi32(bsums, 0xaa)));
        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(s4, _mm256_shuffle_epi32(bsums, 0xff)));
#endif
        isum[0] = _mm256_mullo_epi32(sumi, _mm256_set1_epi32(min));

    } else {
        auto s1 = _mm256_mullo_epi16(_mm256_set1_epi16(min), MM256_SET_M128I(_mm256_extracti128_si256(t3, 0), _mm256_extracti128_si256(t1, 0))); // blocks 0, 1,  8, 9
        auto s2 = _mm256_mullo_epi16(_mm256_set1_epi16(min), MM256_SET_M128I(_mm256_extracti128_si256(t3, 1), _mm256_extracti128_si256(t1, 1))); // blocks 2, 3, 10, 11
        auto s3 = _mm256_mullo_epi16(_mm256_set1_epi16(min), MM256_SET_M128I(_mm256_extracti128_si256(t4, 0), _mm256_extracti128_si256(t2, 0))); // blocks 4, 5, 12, 13
        auto s4 = _mm256_mullo_epi16(_mm256_set1_epi16(min), MM256_SET_M128I(_mm256_extracti128_si256(t4, 1), _mm256_extracti128_si256(t2, 1))); // blocks 6, 7, 14, 15
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto bsums = q8.load_bsums(iy, ibl);
#ifdef HAVE_FANCY_SIMD
            isum[iy] = _mm256_dpwssd_epi32(isum[iy], s1, _mm256_shuffle_epi32(bsums, 0x00));
            isum[iy] = _mm256_dpwssd_epi32(isum[iy], s2, _mm256_shuffle_epi32(bsums, 0x55));
            isum[iy] = _mm256_dpwssd_epi32(isum[iy], s3, _mm256_shuffle_epi32(bsums, 0xaa));
            isum[iy] = _mm256_dpwssd_epi32(isum[iy], s4, _mm256_shuffle_epi32(bsums, 0xff));
#else
            isum[iy] = _mm256_add_epi32(isum[iy], _mm256_madd_epi16(s1, _mm256_shuffle_epi32(bsums, 0x00)));
            isum[iy] = _mm256_add_epi32(isum[iy], _mm256_madd_epi16(s2, _mm256_shuffle_epi32(bsums, 0x55)));
            isum[iy] = _mm256_add_epi32(isum[iy], _mm256_madd_epi16(s3, _mm256_shuffle_epi32(bsums, 0xaa)));
            isum[iy] = _mm256_add_epi32(isum[iy], _mm256_madd_epi16(s4, _mm256_shuffle_epi32(bsums, 0xff)));
#endif
        }
    }
}

template <int nrc_y>
inline void iq2345_k_accum_mins(int ibl, __m256i i8scales1, __m256i i8scales2, const Q8<nrc_y, block_q8_K>& q8, __m256i shuff,
        __m256i extra, __m256i * isum, int8_t min, int8_t delta) {
    auto mask = _mm256_set_epi64x(0x0808080808080808, 0x0404040404040404, 0x0202020202020202, 0x0101010101010101);
    auto vdelta = _mm256_set1_epi8(delta);
    auto vmin   = _mm256_set1_epi8(min);
    auto min1 = _mm256_add_epi8(vmin, _mm256_and_si256(vdelta, _mm256_cmpeq_epi8(_mm256_and_si256(extra, mask), mask)));
    auto min2 = _mm256_add_epi8(vmin, _mm256_and_si256(vdelta, _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_srli_epi16(extra, 4), mask), mask)));
    auto t1 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(i8scales1, 0)), shuff); // blocks  0,  1,  2,  3 for each row
    auto t2 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(i8scales1, 1)), shuff); // blocks  4,  5,  6,  7 for each row
    auto t3 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(i8scales2, 0)), shuff); // blocks  8,  9, 10, 11 for each row
    auto t4 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(i8scales2, 1)), shuff); // blocks 12, 13, 14, 15 for each row
    auto m1 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(min1, 0)), shuff); // blocks  0,  1,  2,  3 for each row
    auto m2 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(min1, 1)), shuff); // blocks  4,  5,  6,  7 for each row
    auto m3 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(min2, 0)), shuff); // blocks  8,  9, 10, 11 for each row
    auto m4 = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(min2, 1)), shuff); // blocks 12, 13, 14, 15 for each row
    auto s1 = _mm256_mullo_epi16(MM256_SET_M128I(_mm256_extracti128_si256(m3, 0), _mm256_extracti128_si256(m1, 0)),
                                 MM256_SET_M128I(_mm256_extracti128_si256(t3, 0), _mm256_extracti128_si256(t1, 0))); // blocks 0, 1,  8, 9
    auto s2 = _mm256_mullo_epi16(MM256_SET_M128I(_mm256_extracti128_si256(m3, 1), _mm256_extracti128_si256(m1, 1)),
                                 MM256_SET_M128I(_mm256_extracti128_si256(t3, 1), _mm256_extracti128_si256(t1, 1))); // blocks 2, 3, 10, 11
    auto s3 = _mm256_mullo_epi16(MM256_SET_M128I(_mm256_extracti128_si256(m4, 0), _mm256_extracti128_si256(m2, 0)),
                                 MM256_SET_M128I(_mm256_extracti128_si256(t4, 0), _mm256_extracti128_si256(t2, 0))); // blocks 4, 5, 12, 13
    auto s4 = _mm256_mullo_epi16(MM256_SET_M128I(_mm256_extracti128_si256(m4, 1), _mm256_extracti128_si256(m2, 1)),
                                 MM256_SET_M128I(_mm256_extracti128_si256(t4, 1), _mm256_extracti128_si256(t2, 1))); // blocks 6, 7, 14, 15
    for (int iy = 0; iy < nrc_y; ++iy) {
        auto bsums = q8.load_bsums(iy, ibl);
#ifdef HAVE_FANCY_SIMD
        isum[iy] = _mm256_dpwssd_epi32(isum[iy], s1, _mm256_shuffle_epi32(bsums, 0x00));
        isum[iy] = _mm256_dpwssd_epi32(isum[iy], s2, _mm256_shuffle_epi32(bsums, 0x55));
        isum[iy] = _mm256_dpwssd_epi32(isum[iy], s3, _mm256_shuffle_epi32(bsums, 0xaa));
        isum[iy] = _mm256_dpwssd_epi32(isum[iy], s4, _mm256_shuffle_epi32(bsums, 0xff));
#else
        isum[iy] = _mm256_add_epi32(isum[iy], _mm256_madd_epi16(s1, _mm256_shuffle_epi32(bsums, 0x00)));
        isum[iy] = _mm256_add_epi32(isum[iy], _mm256_madd_epi16(s2, _mm256_shuffle_epi32(bsums, 0x55)));
        isum[iy] = _mm256_add_epi32(isum[iy], _mm256_madd_epi16(s3, _mm256_shuffle_epi32(bsums, 0xaa)));
        isum[iy] = _mm256_add_epi32(isum[iy], _mm256_madd_epi16(s4, _mm256_shuffle_epi32(bsums, 0xff)));
#endif
    }
}

template <int nrc_y>
static void mul_mat_iq3_k_r4_q8_k(int n, const void * vx, size_t bx, const DataInfo& info, int nrc_x) {
    GGML_ASSERT(nrc_x%4 == 0);
    Q8<nrc_y, block_q8_K> q8(info);
    auto m4 = _mm256_set1_epi8(0xf);
    auto ms  = _mm256_set1_epi8(8);
    auto m03 = _mm256_set1_epi8(0x03);
    auto m04 = _mm256_set1_epi8(0x04);
    auto smask = _mm256_set_epi64x(0x0808080808080808, 0x0404040404040404, 0x0202020202020202, 0x0101010101010101);
    auto shift_shuffle = _mm256_set_epi64x(0x0707070706060606, 0x0505050504040404, 0x0303030302020202, 0x0101010100000000);
    auto values128 = _mm_loadu_si128((const __m128i *)iq3nl_values);
    auto values = MM256_SET_M128I(values128, values128);
    values = _mm256_add_epi8(values, _mm256_set1_epi8(64));
    static const uint8_t k_shuff[32] = {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15};
    auto shuff = _mm256_loadu_si256((const __m256i *)k_shuff);
#ifndef HAVE_FANCY_SIMD
    auto s_shuffle = _mm256_set_epi64x(0x0f0e0f0e0d0c0d0c, 0x0b0a0b0a09080908, 0x0706070605040504, 0x0302030201000100);
#endif
    int nbl = n / QK_K;
    __m256  acc[nrc_y] = {};
    __m256i qx[4];
    uint64_t stored_scales[8];
    for (int ix = 0; ix < nrc_x; ix += 4) {
        const block_iq3_k_r4 * iq3 = (const block_iq3_k_r4 *)((const char *)vx + (ix+0)*bx);
        for (int ibl = 0; ibl < nbl; ++ibl) { // Block of 256
            auto dl = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)iq3[ibl].d));
            auto d4 = _mm256_set_m128(dl, dl);
            auto extra = _mm256_set1_epi64x(*(const uint64_t *)iq3[ibl].extra);
            auto slbits = _mm256_loadu_si256((const __m256i *)iq3[ibl].scales_l);
            auto sl1 = _mm256_add_epi8(_mm256_slli_epi16(_mm256_and_si256(slbits, m4), 1), _mm256_set1_epi8(1));
            auto sl2 = _mm256_add_epi8(_mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(slbits, 4), m4), 1), _mm256_set1_epi8(1));
            auto sh = _mm256_set1_epi64x(((const uint64_t *)iq3[ibl].scales_h)[0]);
            auto sh1 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(sh, smask), smask), _mm256_set1_epi8(1));
            auto sh2 = _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(_mm256_srli_epi16(sh, 4), smask), smask), _mm256_set1_epi8(1));
            auto i8scales1 = _mm256_sign_epi8(sl1, sh1);
            auto i8scales2 = _mm256_sign_epi8(sl2, sh2);
            _mm256_storeu_si256((__m256i *)stored_scales+0, i8scales1);
            _mm256_storeu_si256((__m256i *)stored_scales+1, i8scales2);
            __m256i isum[nrc_y] = {};
            iq234_k_accum_mins(ibl, i8scales1, i8scales2, q8, shuff, isum, -64);
            for (int ib = 0; ib < QK_K/32; ++ib) {
#ifdef HAVE_FANCY_SIMD
                auto scales = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(stored_scales + ib)));
#else
                auto scales = _mm256_shuffle_epi8(_mm256_cvtepi8_epi16(_mm_set1_epi64x(stored_scales[ib])), s_shuffle);
#endif
                auto lb = _mm256_loadu_si256((const __m256i *)iq3[ibl].qs+ib);
                auto hbits = _mm_loadu_si128((const __m128i *)iq3[ibl].qh+ib);
                auto hb = MM256_SET_M128I(hbits, _mm_slli_epi16(hbits, 4));
                auto shift = _mm256_and_si256(ms, _mm256_slli_epi16(extra, 3)); extra = _mm256_srli_epi16(extra, 1);
                shift = _mm256_shuffle_epi8(shift, shift_shuffle);
                qx[0] = _mm256_or_si256(_mm256_and_si256(lb, m03),                       _mm256_and_si256(m04, _mm256_srli_epi16(hb, 2)));
                qx[1] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 2), m03), _mm256_and_si256(m04, _mm256_srli_epi16(hb, 3)));
                qx[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 4), m03), _mm256_and_si256(m04, _mm256_srli_epi16(hb, 4)));
                qx[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 6), m03), _mm256_and_si256(m04, _mm256_srli_epi16(hb, 5)));
                qx[0] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[0], shift));
                qx[1] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[1], shift));
                qx[2] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[2], shift));
                qx[3] = _mm256_shuffle_epi8(values, _mm256_add_epi8(qx[3], shift));
                for (int iy = 0; iy < nrc_y; ++iy) {
                    auto y = _mm256_loadu_si256((const __m256i*)q8.y[iy][ibl].qs+ib);
#ifdef HAVE_FANCY_SIMD
                    auto sumi = _mm256_setzero_si256();
                    sumi = _mm256_dpbusd_epi32(sumi, qx[0], _mm256_shuffle_epi32(y, 0x00));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[1], _mm256_shuffle_epi32(y, 0x55));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[2], _mm256_shuffle_epi32(y, 0xaa));
                    sumi = _mm256_dpbusd_epi32(sumi, qx[3], _mm256_shuffle_epi32(y, 0xff));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_mullo_epi32(scales, sumi));
#else
                    auto sumi1 = _mm256_maddubs_epi16(qx[0], _mm256_shuffle_epi32(y, 0x00));
                    auto sumi2 = _mm256_maddubs_epi16(qx[1], _mm256_shuffle_epi32(y, 0x55));
                    auto sumi3 = _mm256_maddubs_epi16(qx[2], _mm256_shuffle_epi32(y, 0xaa));
                    auto sumi4 = _mm256_maddubs_epi16(qx[3], _mm256_shuffle_epi32(y, 0xff));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_add_epi32(_mm256_madd_epi16(scales, sumi1), _mm256_madd_epi16(scales, sumi2)));
                    isum[iy] = _mm256_add_epi32(isum[iy], _mm256_add_epi32(_mm256_madd_epi16(scales, sumi3), _mm256_madd_epi16(scales, sumi4)));
#endif
                }
            }
            for (int iy = 0; iy < nrc_y; ++iy) {
                acc[iy] = _mm256_fmadd_ps(_mm256_mul_ps(d4, _mm256_set1_ps(q8.scale(iy, ibl))), _mm256_cvtepi32_ps(isum[iy]), acc[iy]);
            }
        }
        for (int iy = 0; iy < nrc_y; ++iy) {
            auto sum = _mm_add_ps(_mm256_castps256_ps128(acc[iy]), _mm256_extractf128_ps(acc[iy], 1));
            acc[iy] = _mm256_setzero_ps();
            info.store(ix+0, iy, sum);
        }
    }
}

} // namespace

bool iqk_convert_iqk_quants_q80_r8(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    if (n%QK_K != 0 || nrc_x%8 != 0) return false;
    switch (ggml_type(type)) {
        case GGML_TYPE_IQ2_KS : iqk_convert_iq2_ks_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ2_K  : iqk_convert_iq2_k_q8_k_r8 (n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ2_KL : iqk_convert_iq2_kl_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ3_KS : iqk_convert_iq3_ks_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ3_K  : iqk_convert_iq3_k_q8_k_r8 (n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ4_KSS: iqk_convert_iq4_kss_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ4_KS : iqk_convert_iq4_ks_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ4_K  : iqk_convert_iq4_k_q8_k_r8 (n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ5_KS : iqk_convert_iq5_ks_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ5_K  : iqk_convert_iq5_k_q8_k_r8 (n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ6_K  : iqk_convert_iq6_k_q8_k_r8 (n, vx, bx, vy, nrc_x); break;
        default: return false;
    }
    return true;
}

bool iqk_set_kernels_iqk_quants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, mul_mat_t& func16) {

    auto etypeA = ggml_type(typeA);
    auto expected_type_B = etypeA == GGML_TYPE_IQ4_KS_R4 || etypeA == GGML_TYPE_IQ5_KS_R4 ? GGML_TYPE_Q8_K32 : GGML_TYPE_Q8_K;
    if (ne00%QK_K != 0 || ggml_type(typeB) != expected_type_B) {
        return false;
    }

    func16 = nullptr;

    switch (typeA) {
        case GGML_TYPE_IQ2_KS:
            set_functions<DequantizerIQ2KS>(kernels);
            break;
        case GGML_TYPE_IQ2_K:
            set_functions<DequantizerIQ2K>(kernels);
            break;
        case GGML_TYPE_IQ2_KL:
            set_functions<DequantizerIQ2KL>(kernels);
#ifdef HAVE_FANCY_SIMD
            func16 = mul_mat_iqX_k_q8_K_AVX512_new<DequantizerIQ2KL, 16>;
#endif
            break;
        case GGML_TYPE_IQ3_KS:
            set_functions<DequantizerIQ3KS>(kernels);
            break;
        case GGML_TYPE_IQ3_K:
            set_functions<DequantizerIQ3K>(kernels);
            break;
        case GGML_TYPE_IQ4_KSS:
            set_functions<DequantizerIQ4KSS>(kernels);
            break;
       case GGML_TYPE_IQ4_KS:
            set_functions<DequantizerIQ4KS>(kernels);
            break;
        case GGML_TYPE_IQ4_K:
            set_functions<DequantizerIQ4K>(kernels);
            break;
        case GGML_TYPE_IQ5_KS:
            set_functions<DequantizerIQ5KS>(kernels);
            break;
        case GGML_TYPE_IQ5_K:
            set_functions<DequantizerIQ5K>(kernels);
            break;
        case GGML_TYPE_IQ6_K:
            set_functions<DequantizerIQ6K>(kernels);
            break;
        case GGML_TYPE_IQ2_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq2_k_r4_q8_k, kernels);
            break;
        case GGML_TYPE_IQ3_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq3_k_r4_q8_k, kernels);
#ifdef HAVE_FANCY_SIMD
            func16 = mul_mat_iq3_k_r4_q8_k<16>;
#endif
            break;
        case GGML_TYPE_IQ4_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq4_k_r4_q8_k, kernels);
            func16  = mul_mat_iq4_k_r4_q8_k<16>;
            break;
        case GGML_TYPE_IQ4_KS_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq4_ks_r4_q8_k, kernels);
#ifndef HAVE_FANCY_SIMD
            // For some reason Zen4 does not like this particular function
            func16 = mul_mat_iq4_ks_r4_q8_k<16>;
#endif
            break;
        case GGML_TYPE_IQ5_KS_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq5_ks_r4_q8_k, kernels);
#ifndef HAVE_FANCY_SIMD
            // For some reason Zen4 does not like this particular function
            func16 = mul_mat_iq5_ks_r4_q8_k<16>;
#endif
            break;
        case GGML_TYPE_IQ5_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq5_k_r4_q8_k, kernels);
            func16  = mul_mat_iq5_k_r4_q8_k<16>;
            break;
        default:
            return false;
    }

    return true;

}

#else // ----------------------------------------- __aarch64__ ---------------------------------------------
namespace {

inline int32x4x4_t make_wider(const int16x8x2_t& scales16) {
    int32x4x4_t scales = {
        vmovl_s16(vget_low_s16 (scales16.val[0])),
        vmovl_s16(vget_high_s16(scales16.val[0])),
        vmovl_s16(vget_low_s16 (scales16.val[1])),
        vmovl_s16(vget_high_s16(scales16.val[1])),
    };
    return scales;
}

inline int32x4x4_t make_wider_8(const int8x16_t& scales8) {
    int16x8x2_t scales16{vmovl_s8(vget_low_s8(scales8)), vmovl_s8(vget_high_s8(scales8))};
    return make_wider(scales16);
}

template <typename Q8>
inline void accum_mins_16(const int16x8x2_t& mins, const Q8& q8, float32x4_t * acc, int i, float c) {
    for (int iy = 0; iy < Q8::nrc_y; ++iy) {
        auto q8s = q8.load_bsums(iy, i);
        int32x4_t b1 = vmull_s16(vget_low_s16 (mins.val[0]), vget_low_s16 (q8s.val[0]));
        int32x4_t b2 = vmull_s16(vget_high_s16(mins.val[0]), vget_high_s16(q8s.val[0]));
        int32x4_t b3 = vmull_s16(vget_low_s16 (mins.val[1]), vget_low_s16 (q8s.val[1]));
        int32x4_t b4 = vmull_s16(vget_high_s16(mins.val[1]), vget_high_s16(q8s.val[1]));
        float32x4_t prod = vcvtq_f32_s32(vaddq_s32(vaddq_s32(b1, b2), vaddq_s32(b3, b4)));
        acc[iy] = vmlaq_f32(acc[iy], prod, vdupq_n_f32(c*q8.scale(iy, i)));
    }
}

struct Scale16Extra {
    template <typename Q8>
    static inline int32x4x4_t new_block(int i, float d, uint16_t extra, uint8_t val,
            const int8x16_t& scales8, const Q8& q8, float32x4_t * acc) {
        uint8x16_t e8 = vreinterpretq_u8_u16(vdupq_n_u16(extra));
        e8 = vceqq_u8(vandq_u8(e8, emask), emask);
        e8 = vqtbl1q_u8(vandq_u8(e8, vdupq_n_u8(val)), eshuff);
        int16x8x2_t extra16 = {vmull_s8(vget_low_s8 (e8), vget_low_s8 (scales8)),
                               vmull_s8(vget_high_s8(e8), vget_high_s8(scales8))};
        accum_mins_16(extra16, q8, acc, i, d);
        return make_wider_8(scales8);
    }

    constexpr static uint32x4_t emask  = {0x02020101, 0x08080404, 0x20201010, 0x80804040};
    constexpr static uint32x4_t eshuff = {0x06040200, 0x0e0c0a08, 0x07050301, 0x0f0d0b09};
};

struct DequantizerIQ3KS final : public BaseDequantizer<block_iq3_ks, true, true> {

    DequantizerIQ3KS(const void * vx, size_t bx, int nrc) : BaseDequantizer(vx, bx, nrc), values(load_values()) {}

    constexpr static int num_blocks() { return 8; }
    constexpr static bool should_scale_quants() { return false; }

    template <typename Q8>
    inline int32x4x2_t new_block(int i, const Q8& q8, float32x4_t * acc) {
        (void)q8;
        (void)acc;
        uint32_t aux32; std::memcpy(&aux32, x[i].scales, 4);
        auto scl8 = vand_s8(vreinterpret_s8_u32(uint32x2_t{aux32, aux32 >> 4}), vdup_n_s8(0xf));
        auto sch8 = vdup_n_u8(x[i].extra & 0xff);
        sch8 = vand_u8(vceq_u8(vand_u8(sch8, shmask), shmask), vdup_n_u8(16));
        scl8 = vsub_s8(vadd_s8(scl8, vreinterpret_s8_u8(sch8)), vdup_n_s8(16));
        auto scales16 = vmovl_s8(scl8);
        int32x4x2_t scales = {vmovl_s16(vget_low_s16(scales16)), vmovl_s16(vget_high_s16(scales16))};
        return scales;
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs+32*j);
        if (j == 0) {
            hbits = vld1q_u8_x2(x[i].qh);
        }
        else {
            hbits.val[0] = vshrq_n_u8(hbits.val[0], 4);
            hbits.val[1] = vshrq_n_u8(hbits.val[1], 4);
        }
        uint8_t extra = x[i].extra >> (8 + 4*j);
        bits.b1.val[0] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(bits.b1.val[0], vandq_u8(vshlq_n_u8(hbits.val[0], 2), hmask)));
        bits.b1.val[1] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(bits.b1.val[1], vandq_u8(vshlq_n_u8(hbits.val[1], 2), hmask))); extra >>= 1;
        bits.b1.val[2] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(bits.b1.val[2], vandq_u8(vshlq_n_u8(hbits.val[0], 1), hmask)));
        bits.b1.val[3] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(bits.b1.val[3], vandq_u8(vshlq_n_u8(hbits.val[1], 1), hmask))); extra >>= 1;
        bits.b2.val[0] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(bits.b2.val[0], vandq_u8(hbits.val[0], hmask)));
        bits.b2.val[1] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(bits.b2.val[1], vandq_u8(hbits.val[1], hmask))); extra >>= 1;
        bits.b2.val[2] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(bits.b2.val[2], vandq_u8(vshrq_n_u8(hbits.val[0], 1), hmask)));
        bits.b2.val[3] = vqtbl1q_s8(values.val[extra & 1], vorrq_u8(bits.b2.val[3], vandq_u8(vshrq_n_u8(hbits.val[1], 1), hmask)));
    }
    static int8x16x2_t load_values() {
        auto v1 = vld1_s8(iq3nl_values + 0);
        auto v2 = vld1_s8(iq3nl_values + 8);
        return { vcombine_s8(v1, v1), vcombine_s8(v2, v2) };
    }

    Q2bits bits;
    uint8x16x2_t hbits;
    const int8x16x2_t values;
    const uint8x16_t hmask = vdupq_n_u8(4);
    const uint8x8_t shmask = vreinterpret_u8_u64(vdup_n_u64(0x8040201008040201));
};

void iqk_convert_iq3_ks_q8_k_r8(int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc_x%8 == 0);

    int nb = n/QK_K;

    const block_iq3_ks * x8[8];

    block_q8_k_r8 * y = (block_q8_k_r8 *)vy;

    int8x16x2_t values;
    {
        auto v1 = vld1_s8(iq3nl_values+0);
        auto v2 = vld1_s8(iq3nl_values+8);
        values.val[0] = vcombine_s8(v1, v1);
        values.val[1] = vcombine_s8(v2, v2);
    }

    ggml_half dh[8];
    int8x16x2_t xv[8];
    uint32_t block[8];
    int8_t   ls[16];

    auto ml = vdupq_n_u8(0x03);
    auto mh = vdupq_n_u8(0x04);

    for (int ix = 0; ix < nrc_x; ix += 8) {
        for (int k = 0; k < 8; ++k) {
            auto dptr = (const ggml_half *)((const char *)vx + (ix+k)*bx);
            dh[k] = dptr[0];
            x8[k] = (const block_iq3_ks *)(dptr + 1);
        }
        for (int i = 0; i < nb; ++i) {
            for (int k = 0; k < 8; ++k) {
                auto extra = x8[k][i].extra;
                auto extra_v = extra >> 8;
                auto hbits = vld1q_u8_x2(x8[k][i].qh);
                for (int i128 = 0; i128 < 2; ++i128) {

                    ls[8*i128+0] = ls[8*i128+1] = int8_t(((x8[k][i].scales[0] >> 4*i128) & 0xf) | ((extra << 4) & 0x10)) - 16;
                    ls[8*i128+2] = ls[8*i128+3] = int8_t(((x8[k][i].scales[1] >> 4*i128) & 0xf) | ((extra << 3) & 0x10)) - 16;
                    ls[8*i128+4] = ls[8*i128+5] = int8_t(((x8[k][i].scales[2] >> 4*i128) & 0xf) | ((extra << 2) & 0x10)) - 16;
                    ls[8*i128+6] = ls[8*i128+7] = int8_t(((x8[k][i].scales[3] >> 4*i128) & 0xf) | ((extra << 1) & 0x10)) - 16;

                    auto bits = vld1q_u8_x2(x8[k][i].qs+32*i128);
                    xv[4*i128+0].val[0] = vqtbl1q_s8(values.val[extra_v & 1], vorrq_u8(vandq_u8(bits.val[0], ml), vandq_u8(vshlq_n_u8(hbits.val[0], 2), mh)));
                    xv[4*i128+0].val[1] = vqtbl1q_s8(values.val[extra_v & 1], vorrq_u8(vandq_u8(bits.val[1], ml), vandq_u8(vshlq_n_u8(hbits.val[1], 2), mh))); extra_v >>= 1;
                    xv[4*i128+1].val[0] = vqtbl1q_s8(values.val[extra_v & 1], vorrq_u8(vandq_u8(vshrq_n_u8(bits.val[0], 2), ml), vandq_u8(vshlq_n_u8(hbits.val[0], 1), mh)));
                    xv[4*i128+1].val[1] = vqtbl1q_s8(values.val[extra_v & 1], vorrq_u8(vandq_u8(vshrq_n_u8(bits.val[1], 2), ml), vandq_u8(vshlq_n_u8(hbits.val[1], 1), mh))); extra_v >>= 1;
                    xv[4*i128+2].val[0] = vqtbl1q_s8(values.val[extra_v & 1], vorrq_u8(vandq_u8(vshrq_n_u8(bits.val[0], 4), ml), vandq_u8(hbits.val[0], mh)));
                    xv[4*i128+2].val[1] = vqtbl1q_s8(values.val[extra_v & 1], vorrq_u8(vandq_u8(vshrq_n_u8(bits.val[1], 4), ml), vandq_u8(hbits.val[1], mh))); extra_v >>= 1;
                    xv[4*i128+3].val[0] = vqtbl1q_s8(values.val[extra_v & 1], vorrq_u8(vshrq_n_u8(bits.val[0], 6), vandq_u8(vshrq_n_u8(hbits.val[0], 1), mh)));
                    xv[4*i128+3].val[1] = vqtbl1q_s8(values.val[extra_v & 1], vorrq_u8(vshrq_n_u8(bits.val[1], 6), vandq_u8(vshrq_n_u8(hbits.val[1], 1), mh))); extra_v >>= 1;
                    hbits.val[0] = vshrq_n_u8(hbits.val[0], 4);
                    hbits.val[1] = vshrq_n_u8(hbits.val[1], 4);
                    extra >>= 4;
                }
                float dnew = convert_to_q8_k_r8(1.f/127, xv, ls, block, (uint32_t *)y[i].qs + k);
                y[i].d[k] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(dh[k])*dnew);
            }
        }
        y += nb;
    }
}

} // namespace

bool iqk_convert_iqk_quants_q80_r8(int type, int n, const void * vx, size_t bx, void * vy, int nrc_x) {
    if (n%QK_K != 0 || nrc_x%8 != 0) return false;
    switch (ggml_type(type)) {
        case GGML_TYPE_IQ2_KS : iqk_convert_iq2_ks_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ2_K  : iqk_convert_iq2_k_q8_k_r8 (n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ2_KL : iqk_convert_iq2_kl_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ3_KS : iqk_convert_iq3_ks_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ3_K  : iqk_convert_iq3_k_q8_k_r8 (n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ4_KSS: iqk_convert_iq4_kss_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ4_KS : iqk_convert_iq4_ks_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ4_K  : iqk_convert_iq4_k_q8_k_r8 (n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ5_KS : iqk_convert_iq5_ks_q8_k_r8(n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ5_K  : iqk_convert_iq5_k_q8_k_r8 (n, vx, bx, vy, nrc_x); break;
        case GGML_TYPE_IQ6_K  : iqk_convert_iq6_k_q8_k_r8 (n, vx, bx, vy, nrc_x); break;
        default: return false;
    }
    return true;
}

bool iqk_set_kernels_iqk_quants(int ne00, int typeA, int typeB, std::array<mul_mat_t, IQK_MAX_NY>& kernels, [[maybe_unused]] mul_mat_t& func16) {

    if (ne00%QK_K != 0 || ggml_type(typeB) != GGML_TYPE_Q8_K) {
        return false;
    }

    func16 = nullptr;

    switch (typeA) {
        case GGML_TYPE_IQ2_KS:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ2KS, kernels);
            break;
        case GGML_TYPE_IQ2_K:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ2K, kernels);
            break;
        case GGML_TYPE_IQ2_KL:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ2KL, kernels);
            break;
        case GGML_TYPE_IQ3_KS:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ3KS, kernels);
            break;
        case GGML_TYPE_IQ3_K:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ3K, kernels);
            break;
        case GGML_TYPE_IQ4_KSS:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ4KSS, kernels);
            break;
       case GGML_TYPE_IQ4_KS:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ4KS, kernels);
            break;
        case GGML_TYPE_IQ4_K:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ4K, kernels);
            break;
        case GGML_TYPE_IQ5_KS:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ5KS, kernels);
            break;
        case GGML_TYPE_IQ5_K:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ5K, kernels);
            break;
        case GGML_TYPE_IQ6_K:
            IQK_SET_MUL_MAT_FUNCTIONS_T(mul_mat_qX_K_q8_K_T, DequantizerIQ6K, kernels);
            break;
        case GGML_TYPE_IQ2_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq2_k_r4_q8_k, kernels);
            break;
        case GGML_TYPE_IQ3_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq3_k_r4_q8_k, kernels);
            break;
        case GGML_TYPE_IQ4_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq4_k_r4_q8_k, kernels);
            break;
        case GGML_TYPE_IQ4_KS_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq4_ks_r4_q8_k, kernels);
            break;
        case GGML_TYPE_IQ5_KS_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq5_ks_r4_q8_k, kernels);
            break;
        case GGML_TYPE_IQ5_K_R4:
            IQK_SET_MUL_MAT_FUNCTIONS(mul_mat_iq5_k_r4_q8_k, kernels);
            break;
        default:
            return false;
    }

    return true;

}

#endif
#endif
