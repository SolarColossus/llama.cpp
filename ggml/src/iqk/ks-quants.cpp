// KS Quant implementation, using code from ik_llama's ik_quantize.cpp
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#define IQK_IMPLEMENT
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP

#include "include/iqk_quantize.hpp"
#include "ggml.h"
#include "../ggml-common.h"
#include "../ggml-impl.h"

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

struct QHelper {
    QHelper(const float * imatrix, int n_per_row, int block_size) : m_imatrix(imatrix),
        m_n_per_row(n_per_row), m_block_size(block_size) {
        if (m_imatrix) {
            m_weight.resize(m_n_per_row);
        }
    }
    const float * row_weights(const float * x) {
        constexpr float kEps  = 1e-9f;
        constexpr float kEps2 = kEps*kEps;
        if (!m_imatrix) return m_imatrix;
        int nblock = m_n_per_row / m_block_size;
        for (int ib = 0; ib < nblock; ++ib) {
            auto wb_in = m_imatrix + ib*m_block_size;
            auto xb = x + ib*m_block_size;
            auto wb = m_weight.data() + ib*m_block_size;
            float sumw2 = 0, sumx2 = 0, sumwx = 0;
            for (int j = 0; j < m_block_size; ++j) {
                wb[j] = wb_in[j];
                sumw2 += wb[j]*wb[j];
                sumx2 += xb[j]*xb[j];
                sumwx += wb[j]*std::abs(xb[j]);
            }
            if (sumw2 > m_block_size*kEps2 && sumx2 > m_block_size*kEps2 && sumwx > m_block_size*kEps2) continue;
            for (int j = 0; j < m_block_size; ++j) {
                wb[j] = kEps;
            }
        }
        return m_weight.data();
    }
    template <typename Func>
    void quantize(int nrows, const float * src, void * dst, int row_size, const Func& qfunc) {
        auto cdst = (char *)dst;
        for (int row = 0; row < nrows; ++row) {
            auto weights = row_weights(src);
            qfunc(src, cdst, m_n_per_row, weights);
            src  += m_n_per_row;
            cdst += row_size;
        }
    }
private:
    const float * m_imatrix;
    const int     m_n_per_row;
    const int     m_block_size;
    std::vector<float> m_weight;
};

// ============================ IQ2_KL functions ======================================
const int8_t iq3nl_index[111] = {
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  8,  8,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  9,
  9,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 10, 10,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 11, 11,  4,  4,  4,  4,
  4,  4,  4,  4,  4,  4, 12,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5, 13, 13,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
  6,  6,  6,  6, 14, 14,  7,  7,  7,  7,  7,  7,  7,  7, 7
};
inline int best_index_iq3nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 111) return ix < 0 ? 0 : 7;
    ix = iq3nl_index[ix];
    return ix < 8 ? ix : x - values[ix-8] < values[ix-7] - x ? ix-8 : ix-7;
}

// ============================ IQ3_KS functions ======================================
static void quantize_row_iq3_ks_impl(const int super_block_size, const int block_size,
        int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    ggml_half * dptr = (ggml_half *)cy;
    block_iq3_ks * y = (block_iq3_ks *)(dptr + 1);

    const int8_t * shifted_values = values + 8;

    float amax_scale = 0;
    float max_scale = 0;

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq3_ks));
        const float * xbl = x + ibl*super_block_size;
        auto scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < block_size; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < 1e-9f) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/values[0] : max/values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            float best = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq3nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq3nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            if (sumq2_p > 0) {
                d = sumqx_p/sumq2_p;
                best = d*sumqx_p;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            bool is_shifted = false;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(values, -al);
                    q = values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
                }
                id = (itry + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(shifted_values, -al);
                    q = shifted_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
                }
            }
            if (is_shifted) y[ibl].extra |= (1 << (8 + ib));
            scales[ib] = d;
            float ascale = std::abs(d);
            if (ascale > amax_scale) {
                amax_scale = ascale; max_scale = d;
            }
        }
    }
    float d = -max_scale/16;
    *dptr = GGML_FP32_TO_FP16(d);
    if (!d) return;
    float id = d ? 1/d : 0.f;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        auto scales = all_scales + (super_block_size/block_size)*ibl;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const int8_t * block_values = (y[ibl].extra >> (8 + ib)) & 0x01 ? shifted_values : values;
            int l = nearest_int(id*scales[ib]);
            l = std::max(-16, std::min(15, l));
            uint8_t ul = l + 16;
            y[ibl].scales[ib%4] |= (ul & 0xf) << 4*(ib/4);
            y[ibl].extra |= (ul >> 4) << ib;
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            auto qs = y[ibl].qs + (ib/4)*block_size;
            auto qh = y[ibl].qh + (ib/8)*block_size;
            for (int j = 0; j < block_size; ++j) {
                uint8_t i = best_index_iq3nl(block_values, idl*xb[j]);
                qs[j] |= ((i &  3) << 2*(ib%4));
                qh[j] |= ((i >> 2) << (ib%8));
                float w = weight[j];
                float q = block_values[i]*l;
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
        }
    }
    if (sumq2 > 0) *dptr = GGML_FP32_TO_FP16(sumqx/sumq2);
}

void quantize_row_iq3_ks_ref(const float * x, block_iq3_ks * y, int64_t k) {
    quantize_iq3_ks(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq3_ks(const float * x, void * y, int64_t k) {
    quantize_iq3_ks(x, (void *)y, 1, k, nullptr);
}

size_t quantize_iq3_ks(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    constexpr int kBlockSize = 32;
    GGML_ASSERT(n_per_row%QK_K == 0);
    float weight[kBlockSize];
    std::vector<float> all_scales(n_per_row/kBlockSize);
    auto row_size = ggml_row_size(GGML_TYPE_IQ3_KS, n_per_row);
    QHelper helper(imatrix, n_per_row, kBlockSize);
    auto q_func = [&all_scales, &weight, block_size = kBlockSize] (const float * x, void * vy, int n_per_row, const float * imatrix) {
        quantize_row_iq3_ks_impl(QK_K, block_size, n_per_row, x, (char *)vy, all_scales.data(), weight, iq3nl_values, imatrix, 5);
    };
    helper.quantize(nrows, src, dst, row_size, q_func);
    return nrows * row_size;
}

void dequantize_row_iq3_ks(const block_iq3_ks * x, float * y, int64_t k) {
    constexpr int kBlockSize = 32;
    static_assert(QK_K/kBlockSize == 8);
    GGML_ASSERT(k%QK_K == 0);
    const ggml_half * dptr = (const ggml_half *)x;
    float d = GGML_FP16_TO_FP32(*dptr);
    x = (const block_iq3_ks *)(dptr + 1);
    float dl[8];
    int nblock = k/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        for (int j = 0; j < 4; ++j) {
            int ls1 = (x[ibl].scales[j] & 0xf) | (((x[ibl].extra >> (j+0)) & 1) << 4);
            int ls2 = (x[ibl].scales[j] >>  4) | (((x[ibl].extra >> (j+4)) & 1) << 4);
            dl[j+0] = d*(ls1 - 16);
            dl[j+4] = d*(ls2 - 16);
        }
        auto qs = x[ibl].qs;
        auto qh = x[ibl].qh;
        for (int i128 = 0; i128 < QK_K/128; ++i128) {
            for (int ib = 0; ib < 4; ++ib) {
                const int8_t * values = iq3nl_values + ((x[ibl].extra >> (8 + (4*i128+ib)) & 1) << 3);
                for (int j = 0; j < kBlockSize; ++j) {
                    y[j] = dl[4*i128 + ib] * values[((qs[j] >> 2*ib) & 3) | (((qh[j] >> (4*i128+ib)) & 1) << 2)];
                }
                y += kBlockSize;
            }
            qs += kBlockSize;
        }
    }
}

void  vec_dot_iq3_ks_q8_k(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if GGML_USE_IQK_MULMAT
    if (iqk_mul_mat(1, 1, n, GGML_TYPE_IQ3_KS, vx, 0, GGML_TYPE_Q8_K, vy, 0, s, 0, 0, 1)) {
        return;
    }
#endif
    GGML_ASSERT(n%QK_K == 0);
    GGML_ASSERT(nrc == 1);
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_ABORT("Not implemented");
}
