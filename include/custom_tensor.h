#pragma once
#include "fp_utils.h"
#include <cstddef>
#include <cstdint>

struct CustomConfig {
    enum class ABPrec    : uint8_t { FP16, BF16, FP8_E5M2, FP8_E4M3, FP4_E2M1 };
    enum class CDPrec    : uint8_t { FP32, FP16 };
    enum class RoundMode : uint8_t { RNE, RTZ };
    enum class ScaleType : uint8_t { UE8M0, UE4M3 };

    ABPrec    ab_prec    = ABPrec::FP16;
    CDPrec    cd_prec    = CDPrec::FP32;
    int       dp_width   = 16;
    int       mant_width = 25;
    RoundMode round_mode = RoundMode::RTZ;
    bool      use_scale  = false;
    int       scale_group = 16;
    ScaleType scale_type = ScaleType::UE8M0;
    int       vec_len    = 0;
};

uint32_t custom_dot_product(const uint32_t* a, const uint32_t* b,
                             const uint8_t* sa, const uint8_t* sb,
                             size_t len, uint32_t c,
                             const CustomConfig& cfg);
