//
// Copyright (C) 2024-2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#pragma once

#include <stdint.h>
#include <stddef.h>

#define GGML_COMMON_DECL_C
#include "../ggml-common.h"

#ifdef __cplusplus
#define GGML_RESTRICT
extern "C" {
#else
#define GGML_RESTRICT restrict
#endif

void   quantize_row_iq3_ks_ref(const float * GGML_RESTRICT x, block_iq3_ks  * GGML_RESTRICT y, int64_t k);
void   quantize_row_iq3_ks(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
size_t quantize_iq3_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
void   dequantize_row_iq3_ks(const block_iq3_ks  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
void   vec_dot_iq3_ks_q8_k(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif
