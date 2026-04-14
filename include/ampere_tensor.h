#pragma once
#include "fp_utils.h"
#include <cstddef>

fp32_t ampere_dp8a_fp32(const fp16_t* a, const fp16_t* b, size_t len, fp32_t c);

fp16_t ampere_dp8a_fp16(const fp16_t* a, const fp16_t* b, size_t len, fp16_t c);

fp32_t ampere_dp8a_bf16(const bf16_t* a, const bf16_t* b, size_t len, fp32_t c);