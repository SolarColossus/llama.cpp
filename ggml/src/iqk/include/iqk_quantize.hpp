//
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include <stdint.h>
#include <stddef.h>

#define GGML_COMMON_DECL_C
#include "../../ggml-common.h"

#ifdef __cplusplus
#define GGML_RESTRICT
extern "C" {
#else
#define GGML_RESTRICT restrict
#endif

void   quantize_row_iq2_k_ref(const float * GGML_RESTRICT x, block_iq2_k  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq2_k(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq2_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq2_k(const block_iq2_k  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq2_k_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq3_k_ref(const float * GGML_RESTRICT x, block_iq3_k  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq3_k(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq3_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq3_k(const block_iq3_k  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq3_k_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq3_ks_ref(const float * GGML_RESTRICT x, block_iq3_ks  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq3_ks(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq3_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq3_ks(const block_iq3_ks  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq3_ks_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq4_k_ref(const float * GGML_RESTRICT x, block_iq4_k  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq4_k(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq4_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq4_k(const block_iq4_k  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq4_k_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq5_k_ref(const float * GGML_RESTRICT x, block_iq5_k  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq5_k(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq5_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq5_k(const block_iq5_k  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq5_k_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_iq6_k_ref(const float * GGML_RESTRICT x, block_iq6_k  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq6_k(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq6_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq6_k(const block_iq6_k  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq6_k_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void   quantize_row_mxfp4_ref(const float * GGML_RESTRICT x, block_mxfp4  * GGML_RESTRICT y, int64_t k);
void   quantize_row_mxfp4(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_mxfp4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_mxfp4(const block_mxfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_mxfp4_q8_0_x4(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

void repack_f32_bf16_r16 (const void * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row);
void repack_bf16_bf16_r16(const void * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row);

void iqk_repack_tensor(struct ggml_tensor * tensor);
bool iqk_modify_tensor(struct ggml_tensor * tensor);

int iqk_repacked_type(const struct ggml_tensor * tensor); // int instead of ggml_type so we don't need to include ggml.h
bool iqk_should_modify_tensor(const struct ggml_tensor * tensor);

// So we can re-pack Microsoft's BitNet I2_S quants
void dequantize_row_ms_i2s(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

typedef void (*to_float_t)  (const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
typedef void (*from_float_t)(const float * GGML_RESTRICT x, void  * GGML_RESTRICT y, int64_t k);
void iqk_quantize_any(int from_type, int to_type,
                      int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                      uint64_t nb0, uint64_t nb1, uint64_t nb2, uint64_t nb3,
                      const void * GGML_RESTRICT x, void * GGML_RESTRICT y, void * work_buffer,
                      to_float_t to_float, from_float_t from_float, int ith, int nth);

bool iqk_validate_tensor(const struct ggml_tensor * src);

#ifdef __cplusplus
}
#endif
