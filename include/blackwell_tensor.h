#pragma once
#include "fp_utils.h"
#include <cstddef>

fp32_t blackwell_dp16a_fp32(const fp16_t* a, const fp16_t* b, size_t len, fp32_t c);
fp16_t blackwell_dp16a_fp16(const fp16_t* a, const fp16_t* b, size_t len, fp16_t c);

fp32_t blackwell_dp16a_bf16(const bf16_t* a, const bf16_t* b, size_t len, fp32_t c);

fp32_t blackwell_dp32a_e5m2_fp32(const e5m2_t* a, const e5m2_t* b, size_t len, fp32_t c);
fp16_t blackwell_dp32a_e5m2_fp16(const e5m2_t* a, const e5m2_t* b, size_t len, fp16_t c);

fp32_t blackwell_dp32a_e4m3_fp32(const e4m3_t* a, const e4m3_t* b, size_t len, fp32_t c);
fp16_t blackwell_dp32a_e4m3_fp16(const e4m3_t* a, const e4m3_t* b, size_t len, fp16_t c);

fp32_t blackwell_dp64a_e2m1_fp32(const e2m1_t* a, const e2m1_t* b, size_t len, fp32_t c);
fp16_t blackwell_dp64a_e2m1_fp16(const e2m1_t* a, const e2m1_t* b, size_t len, fp16_t c);

fp32_t blackwell_mxfp4_e2m1_e8_fp32(const e2m1_t* a, const e2m1_t* b, const uint8_t* sa, const uint8_t* sb, size_t len, fp32_t c);

fp32_t blackwell_nvfp4_e2m1_ue4m3_fp32(const e2m1_t* a, const e2m1_t* b, const uint8_t* sa, const uint8_t* sb, size_t len, fp32_t c);