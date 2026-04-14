#include "hopper_tensor.h"
#include "fp_utils.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  Minimal test harness
// ─────────────────────────────────────────────────────────────────────────────
static int g_pass = 0, g_fail = 0;

static void check(bool cond, const char* expr, const char* file, int line) {
    if (cond) { ++g_pass; }
    else {
        ++g_fail;
        std::printf("  FAIL  %s:%d  %s\n", file, line, expr);
    }
}

#define CHECK(cond)         check((cond), #cond, __FILE__, __LINE__)
#define CHECK_EQ_HEX(a,b)   do { \
    auto _a=(a), _b=(b); \
    if (_a!=_b) { ++g_fail; \
        std::printf("  FAIL  %s:%d  0x%08X != 0x%08X  (%s)\n", \
                    __FILE__,__LINE__,(unsigned)_a,(unsigned)_b, #a " == " #b); \
    } else ++g_pass; } while(0)

#define SECTION(name)  std::printf("\n[%s]\n", name)

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────
static fp16_t h(float f)   { return float_to_fp16(f); }
static bf16_t b(float f)   { return float_to_bf16(f); }
static fp32_t f32(float f) { return float_to_bits(f); }

// ── FP16 constants ────────────────────────────────────────────────────────────
static constexpr fp16_t H_ONE      = 0x3C00u;
static constexpr fp16_t H_TWO      = 0x4000u;
static constexpr fp16_t H_NEG_ONE  = 0xBC00u;
static constexpr fp16_t H_ZERO     = 0x0000u;
static constexpr fp16_t H_NEG_ZERO = 0x8000u;
static constexpr fp16_t H_INF      = 0x7C00u;
static constexpr fp16_t H_NEG_INF  = 0xFC00u;
static constexpr fp16_t H_NAN      = 0x7E00u;
static constexpr fp16_t H_MAX      = 0x7BFFu;   // 65504
static constexpr fp16_t H_SUBNORM1 = 0x0001u;   // 2^(-24)

// ── FP32 constants ────────────────────────────────────────────────────────────
static constexpr fp32_t F_ZERO    = 0x00000000u;
static constexpr fp32_t F_ONE     = 0x3F800000u;
static constexpr fp32_t F_TWO     = 0x40000000u;
static constexpr fp32_t F_NEG_ONE = 0xBF800000u;
static constexpr fp32_t F_INF     = 0x7F800000u;
static constexpr fp32_t F_NEG_INF = 0xFF800000u;
static constexpr fp32_t F_NAN_OUT = 0x7FFFFFFFu;
static constexpr fp16_t H_NAN_OUT = 0x7FFFu;

// ── BF16 constants ────────────────────────────────────────────────────────────
static constexpr bf16_t BF_ONE     = 0x3F80u;
static constexpr bf16_t BF_TWO     = 0x4000u;
static constexpr bf16_t BF_ZERO    = 0x0000u;
static constexpr bf16_t BF_NEG_ONE = 0xBF80u;
static constexpr bf16_t BF_INF     = 0x7F80u;
static constexpr bf16_t BF_NAN     = 0x7FC0u;
static constexpr bf16_t BF_NEG_ZERO= 0x8000u;

// ── E5M2 constants (S EEEEE MM, bias=15) ─────────────────────────────────────
// 1.0 = exp=15, mant=00 → 0 01111 00 = 0x3C
static constexpr e5m2_t E5_ONE      = 0x3Cu;   //  1.0
static constexpr e5m2_t E5_TWO      = 0x40u;   //  2.0 (exp=16, mant=00)
static constexpr e5m2_t E5_ZERO     = 0x00u;   //  0.0
static constexpr e5m2_t E5_NEG_ZERO = 0x80u;   // -0.0
static constexpr e5m2_t E5_NEG_ONE  = 0xBCu;   // -1.0
static constexpr e5m2_t E5_INF      = 0x7Cu;   // +Inf (exp=31, mant=00)
static constexpr e5m2_t E5_NEG_INF  = 0xFCu;   // -Inf (exp=31, mant=00, sign=1)
static constexpr e5m2_t E5_NAN      = 0x7Du;   //  NaN (exp=31, mant=01)
static constexpr e5m2_t E5_SUBNORM1 = 0x01u;   //  smallest positive subnormal
static constexpr e5m2_t E5_MAX      = 0x7Bu;   //  57344 (exp=30, mant=11)

// ── E4M3 constants (S EEEE MMM, bias=7, only 0x7F/0xFF → NaN, no Inf) ────────
// 1.0 = exp=7, mant=000 → 0 0111 000 = 0x38
static constexpr e4m3_t E4_ONE      = 0x38u;   //  1.0
static constexpr e4m3_t E4_TWO      = 0x40u;   //  2.0 (exp=8, mant=000)
static constexpr e4m3_t E4_ZERO     = 0x00u;   //  0.0
static constexpr e4m3_t E4_NEG_ZERO = 0x80u;   // -0.0
static constexpr e4m3_t E4_NEG_ONE  = 0xB8u;   // -1.0
static constexpr e4m3_t E4_NAN      = 0x7Fu;   //  NaN (exp=15)
static constexpr e4m3_t E4_SUBNORM1 = 0x01u;   //  smallest positive subnormal
static constexpr e4m3_t E4_MAX      = 0x77u;   //  240 (exp=14, mant=111) = 2^7×1.875

// ─────────────────────────────────────────────────────────────────────────────
//  Single-element wrappers (rest zero-padded by the public API)
// ─────────────────────────────────────────────────────────────────────────────
static fp32_t dp16_fp32_1(fp16_t a0, fp16_t b0, fp32_t c) {
    return hopper_dp16a_fp32(&a0, &b0, 1, c);
}
static fp16_t dp16_fp16_1(fp16_t a0, fp16_t b0, fp16_t c) {
    return hopper_dp16a_fp16(&a0, &b0, 1, c);
}
static fp32_t dp16_bf16_1(bf16_t a0, bf16_t b0, fp32_t c) {
    return hopper_dp16a_bf16(&a0, &b0, 1, c);
}
static fp32_t dp32_e5m2_fp32_1(e5m2_t a0, e5m2_t b0, fp32_t c) {
    return hopper_dp32a_e5m2_fp32(&a0, &b0, 1, c);
}
static fp16_t dp32_e5m2_fp16_1(e5m2_t a0, e5m2_t b0, fp16_t c) {
    return hopper_dp32a_e5m2_fp16(&a0, &b0, 1, c);
}
static fp32_t dp32_e4m3_fp32_1(e4m3_t a0, e4m3_t b0, fp32_t c) {
    return hopper_dp32a_e4m3_fp32(&a0, &b0, 1, c);
}
static fp16_t dp32_e4m3_fp16_1(e4m3_t a0, e4m3_t b0, fp16_t c) {
    return hopper_dp32a_e4m3_fp16(&a0, &b0, 1, c);
}

// ─────────────────────────────────────────────────────────────────────────────
//  DP16A FP32 accumulator – FP16 inputs
// ─────────────────────────────────────────────────────────────────────────────
static void test_dp16a_fp32_basic() {
    SECTION("DP16A FP32 acc / FP16 inputs – basic");

    CHECK_EQ_HEX(dp16_fp32_1(H_ONE, H_ONE, F_ZERO), F_ONE);
    CHECK_EQ_HEX(dp16_fp32_1(H_ONE, H_ONE, F_ONE),  F_TWO);
    CHECK_EQ_HEX(dp16_fp32_1(H_NEG_ONE, H_ONE, F_ZERO), F_NEG_ONE);
    CHECK_EQ_HEX(dp16_fp32_1(H_NEG_ZERO, H_ONE, F_ZERO), F_ZERO);

    // 2×3 + 4 = 10
    {
        fp16_t a[1]={h(2.0f)}, bv[1]={h(3.0f)};
        CHECK_EQ_HEX(hopper_dp16a_fp32(a, bv, 1, f32(4.0f)), f32(10.0f));
    }

    // 16 × (1×1) + 0 = 16
    {
        fp16_t a[16], bv[16];
        for (int i = 0; i < 16; ++i) { a[i] = H_ONE; bv[i] = H_ONE; }
        CHECK_EQ_HEX(hopper_dp16a_fp32(a, bv, 16, F_ZERO), f32(16.0f));
    }

    // dot([1..16],[1..16]) = 1+4+9+...+256 = 1496
    {
        fp16_t a[16], bv[16];
        for (int i = 0; i < 16; ++i) {
            a[i]  = h((float)(i+1));
            bv[i] = h((float)(i+1));
        }
        CHECK_EQ_HEX(hopper_dp16a_fp32(a, bv, 16, F_ZERO), f32(1496.0f));
    }
}

static void test_dp16a_fp32_specials() {
    SECTION("DP16A FP32 acc / FP16 inputs – specials");

    CHECK_EQ_HEX(dp16_fp32_1(H_NAN,  H_ONE,  F_ZERO),    F_NAN_OUT);
    CHECK_EQ_HEX(dp16_fp32_1(H_ONE,  H_NAN,  F_ZERO),    F_NAN_OUT);
    CHECK_EQ_HEX(dp16_fp32_1(H_ZERO, H_INF,  F_ZERO),    F_NAN_OUT);
    CHECK_EQ_HEX(dp16_fp32_1(H_INF,  H_ZERO, F_ZERO),    F_NAN_OUT);
    CHECK_EQ_HEX(dp16_fp32_1(H_INF,  H_ONE,  F_NEG_INF), F_NAN_OUT);
    CHECK_EQ_HEX(dp16_fp32_1(H_ONE,  H_ONE,  F_NAN_OUT), F_NAN_OUT);
    CHECK_EQ_HEX(dp16_fp32_1(H_INF,  H_ONE,  F_ZERO),    F_INF);
    CHECK_EQ_HEX(dp16_fp32_1(H_INF,  H_NEG_ONE, F_ZERO), F_NEG_INF);
}

static void test_dp16a_fp32_subnorm() {
    SECTION("DP16A FP32 acc / FP16 inputs – subnormals");

    // H_SUBNORM1 = 2^(-24); product with 1 = 2^(-24), a valid FP32 normal
    {
        fp32_t r = dp16_fp32_1(H_SUBNORM1, H_ONE, F_ZERO);
        CHECK(r != F_ZERO);
    }

    // subnormal × subnormal: 2^(-24) × 2^(-24) = 2^(-48), FP32 biased exp=79
    CHECK_EQ_HEX(dp16_fp32_1(H_SUBNORM1, H_SUBNORM1, F_ZERO), (fp32_t)0x27800000u);
}

static void test_dp16a_fp32_chaining() {
    SECTION("DP16A FP32 acc / FP16 inputs – chaining");

    // 32 × (1×1) = 32  (two groups of 16)
    {
        fp16_t a[32], bv[32];
        for (int i = 0; i < 32; ++i) { a[i] = H_ONE; bv[i] = H_ONE; }
        CHECK_EQ_HEX(hopper_dp16a_fp32(a, bv, 32, F_ZERO), f32(32.0f));
    }

    // Length 17 (one full group + partial of 1)
    {
        fp16_t a[17], bv[17];
        for (int i = 0; i < 17; ++i) { a[i] = H_ONE; bv[i] = H_ONE; }
        CHECK_EQ_HEX(hopper_dp16a_fp32(a, bv, 17, F_ZERO), f32(17.0f));
    }

    // Length 7 (partial group only): 1²+2²+...+7² = 140
    {
        fp16_t a[7], bv[7];
        for (int i = 0; i < 7; ++i) { a[i] = h((float)(i+1)); bv[i] = h((float)(i+1)); }
        CHECK_EQ_HEX(hopper_dp16a_fp32(a, bv, 7, F_ZERO), f32(140.0f));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  DP16A FP16 accumulator – FP16 inputs
// ─────────────────────────────────────────────────────────────────────────────
static void test_dp16a_fp16_basic() {
    SECTION("DP16A FP16 acc / FP16 inputs – basic");

    CHECK_EQ_HEX(dp16_fp16_1(H_ONE, H_ONE, H_ZERO),   H_ONE);
    CHECK_EQ_HEX(dp16_fp16_1(H_ONE, H_ONE, H_ONE),    H_TWO);
    CHECK_EQ_HEX(dp16_fp16_1(H_NEG_ONE, H_ONE, H_ZERO), H_NEG_ONE);
    CHECK_EQ_HEX(dp16_fp16_1(H_NEG_ZERO, H_ONE, H_ZERO), H_ZERO);

    // 16 × (1×1) + 0 = 16
    {
        fp16_t a[16], bv[16];
        for (int i = 0; i < 16; ++i) { a[i] = H_ONE; bv[i] = H_ONE; }
        CHECK_EQ_HEX(hopper_dp16a_fp16(a, bv, 16, H_ZERO), h(16.0f));
    }
}

static void test_dp16a_fp16_specials() {
    SECTION("DP16A FP16 acc / FP16 inputs – specials");

    CHECK_EQ_HEX(dp16_fp16_1(H_NAN,  H_ONE,  H_ZERO),    H_NAN_OUT);
    CHECK_EQ_HEX(dp16_fp16_1(H_ZERO, H_INF,  H_ZERO),    H_NAN_OUT);
    CHECK_EQ_HEX(dp16_fp16_1(H_INF,  H_ONE,  H_NEG_INF), H_NAN_OUT);
    CHECK_EQ_HEX(dp16_fp16_1(H_ONE,  H_ONE,  H_NAN_OUT), H_NAN_OUT);
    CHECK_EQ_HEX(dp16_fp16_1(H_INF,  H_ONE,  H_ZERO),    H_INF);
    // H_MAX × H_MAX overflows FP16 → +Inf
    CHECK_EQ_HEX(dp16_fp16_1(H_MAX, H_MAX, H_ZERO), H_INF);
}

static void test_dp16a_fp16_chaining() {
    SECTION("DP16A FP16 acc / FP16 inputs – chaining");

    // 32 × (1×1) = 32
    {
        fp16_t a[32], bv[32];
        for (int i = 0; i < 32; ++i) { a[i] = H_ONE; bv[i] = H_ONE; }
        CHECK_EQ_HEX(hopper_dp16a_fp16(a, bv, 32, H_ZERO), h(32.0f));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  DP16A FP32 accumulator – BF16 inputs
// ─────────────────────────────────────────────────────────────────────────────
static void test_dp16a_bf16_basic() {
    SECTION("DP16A FP32 acc / BF16 inputs – basic");

    CHECK_EQ_HEX(dp16_bf16_1(BF_ONE, BF_ONE, F_ZERO), F_ONE);
    CHECK_EQ_HEX(dp16_bf16_1(BF_ONE, BF_ONE, F_ONE),  F_TWO);
    CHECK_EQ_HEX(dp16_bf16_1(BF_NEG_ONE, BF_ONE, F_ZERO), F_NEG_ONE);
    CHECK_EQ_HEX(dp16_bf16_1(BF_NEG_ZERO, BF_ONE, F_ZERO), F_ZERO);

    // 2×3 + 4 = 10
    {
        bf16_t a[1]={b(2.0f)}, bv[1]={b(3.0f)};
        CHECK_EQ_HEX(hopper_dp16a_bf16(a, bv, 1, f32(4.0f)), f32(10.0f));
    }

    // 16 × (1×1) + 0 = 16
    {
        bf16_t a[16], bv[16];
        for (int i = 0; i < 16; ++i) { a[i] = BF_ONE; bv[i] = BF_ONE; }
        CHECK_EQ_HEX(hopper_dp16a_bf16(a, bv, 16, F_ZERO), f32(16.0f));
    }
}

static void test_dp16a_bf16_specials() {
    SECTION("DP16A FP32 acc / BF16 inputs – specials");

    CHECK_EQ_HEX(dp16_bf16_1(BF_NAN,  BF_ONE, F_ZERO),    F_NAN_OUT);
    CHECK_EQ_HEX(dp16_bf16_1(BF_ZERO, BF_INF, F_ZERO),    F_NAN_OUT);
    CHECK_EQ_HEX(dp16_bf16_1(BF_INF,  BF_ONE, F_NEG_INF), F_NAN_OUT);
    CHECK_EQ_HEX(dp16_bf16_1(BF_ONE,  BF_ONE, F_NAN_OUT), F_NAN_OUT);
    CHECK_EQ_HEX(dp16_bf16_1(BF_INF,  BF_ONE, F_ZERO),    F_INF);
}

static void test_dp16a_bf16_chaining() {
    SECTION("DP16A FP32 acc / BF16 inputs – chaining");

    // 32 × (1×1) = 32
    {
        bf16_t a[32], bv[32];
        for (int i = 0; i < 32; ++i) { a[i] = BF_ONE; bv[i] = BF_ONE; }
        CHECK_EQ_HEX(hopper_dp16a_bf16(a, bv, 32, F_ZERO), f32(32.0f));
    }

    // Length 5 (partial): 5 × (2×2) = 20
    {
        bf16_t a[5], bv[5];
        for (int i = 0; i < 5; ++i) { a[i] = BF_TWO; bv[i] = BF_TWO; }
        CHECK_EQ_HEX(hopper_dp16a_bf16(a, bv, 5, F_ZERO), f32(20.0f));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  DP32A FP32 accumulator – E5M2 inputs
// ─────────────────────────────────────────────────────────────────────────────
static void test_dp32a_e5m2_fp32_basic() {
    SECTION("DP32A FP32 acc / E5M2 inputs – basic");

    CHECK_EQ_HEX(dp32_e5m2_fp32_1(E5_ONE, E5_ONE, F_ZERO),  F_ONE);
    CHECK_EQ_HEX(dp32_e5m2_fp32_1(E5_ONE, E5_ONE, F_ONE),   F_TWO);
    CHECK_EQ_HEX(dp32_e5m2_fp32_1(E5_NEG_ONE, E5_ONE, F_ZERO), F_NEG_ONE);
    CHECK_EQ_HEX(dp32_e5m2_fp32_1(E5_NEG_ZERO, E5_ONE, F_ZERO), F_ZERO);

    // 2×2 + 4 = 8
    {
        e5m2_t a[1]={E5_TWO}, bv[1]={E5_TWO};
        CHECK_EQ_HEX(hopper_dp32a_e5m2_fp32(a, bv, 1, f32(4.0f)), f32(8.0f));
    }

    // 32 × (1×1) + 0 = 32
    {
        e5m2_t a[32], bv[32];
        for (int i = 0; i < 32; ++i) { a[i] = E5_ONE; bv[i] = E5_ONE; }
        CHECK_EQ_HEX(hopper_dp32a_e5m2_fp32(a, bv, 32, F_ZERO), f32(32.0f));
    }
}

static void test_dp32a_e5m2_fp32_specials() {
    SECTION("DP32A FP32 acc / E5M2 inputs – specials");

    CHECK_EQ_HEX(dp32_e5m2_fp32_1(E5_NAN,  E5_ONE,  F_ZERO),    F_NAN_OUT);
    CHECK_EQ_HEX(dp32_e5m2_fp32_1(E5_ONE,  E5_NAN,  F_ZERO),    F_NAN_OUT);
    CHECK_EQ_HEX(dp32_e5m2_fp32_1(E5_ZERO, E5_INF,  F_ZERO),    F_NAN_OUT);
    CHECK_EQ_HEX(dp32_e5m2_fp32_1(E5_INF,  E5_ZERO, F_ZERO),    F_NAN_OUT);
    CHECK_EQ_HEX(dp32_e5m2_fp32_1(E5_INF,  E5_ONE,  F_NEG_INF), F_NAN_OUT);
    CHECK_EQ_HEX(dp32_e5m2_fp32_1(E5_ONE,  E5_ONE,  F_NAN_OUT), F_NAN_OUT);
    CHECK_EQ_HEX(dp32_e5m2_fp32_1(E5_INF,  E5_ONE,  F_ZERO),    F_INF);
    CHECK_EQ_HEX(dp32_e5m2_fp32_1(E5_INF,  E5_NEG_ONE, F_ZERO), F_NEG_INF);
}

static void test_dp32a_e5m2_fp32_subnorm() {
    SECTION("DP32A FP32 acc / E5M2 inputs – subnormals");

    // smallest E5M2 subnormal × 1 should be non-zero in FP32 accumulator
    {
        fp32_t r = dp32_e5m2_fp32_1(E5_SUBNORM1, E5_ONE, F_ZERO);
        CHECK(r != F_ZERO);
    }
}

static void test_dp32a_e5m2_fp32_chaining() {
    SECTION("DP32A FP32 acc / E5M2 inputs – chaining");

    // 64 × (1×1) = 64  (two groups of 32)
    {
        e5m2_t a[64], bv[64];
        for (int i = 0; i < 64; ++i) { a[i] = E5_ONE; bv[i] = E5_ONE; }
        CHECK_EQ_HEX(hopper_dp32a_e5m2_fp32(a, bv, 64, F_ZERO), f32(64.0f));
    }

    // Length 33 (one full + partial of 1)
    {
        e5m2_t a[33], bv[33];
        for (int i = 0; i < 33; ++i) { a[i] = E5_ONE; bv[i] = E5_ONE; }
        CHECK_EQ_HEX(hopper_dp32a_e5m2_fp32(a, bv, 33, F_ZERO), f32(33.0f));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  DP32A FP16 accumulator – E5M2 inputs
// ─────────────────────────────────────────────────────────────────────────────
static void test_dp32a_e5m2_fp16_basic() {
    SECTION("DP32A FP16 acc / E5M2 inputs – basic");

    CHECK_EQ_HEX(dp32_e5m2_fp16_1(E5_ONE, E5_ONE, H_ZERO),    H_ONE);
    CHECK_EQ_HEX(dp32_e5m2_fp16_1(E5_ONE, E5_ONE, H_ONE),     H_TWO);
    CHECK_EQ_HEX(dp32_e5m2_fp16_1(E5_NEG_ONE, E5_ONE, H_ZERO), H_NEG_ONE);
    CHECK_EQ_HEX(dp32_e5m2_fp16_1(E5_NEG_ZERO, E5_ONE, H_ZERO), H_ZERO);

    // 32 × (1×1) = 32
    {
        e5m2_t a[32], bv[32];
        for (int i = 0; i < 32; ++i) { a[i] = E5_ONE; bv[i] = E5_ONE; }
        CHECK_EQ_HEX(hopper_dp32a_e5m2_fp16(a, bv, 32, H_ZERO), h(32.0f));
    }
}

static void test_dp32a_e5m2_fp16_specials() {
    SECTION("DP32A FP16 acc / E5M2 inputs – specials");

    CHECK_EQ_HEX(dp32_e5m2_fp16_1(E5_NAN,  E5_ONE,  H_ZERO),    H_NAN_OUT);
    CHECK_EQ_HEX(dp32_e5m2_fp16_1(E5_ZERO, E5_INF,  H_ZERO),    H_NAN_OUT);
    CHECK_EQ_HEX(dp32_e5m2_fp16_1(E5_INF,  E5_ONE,  H_NEG_INF), H_NAN_OUT);
    CHECK_EQ_HEX(dp32_e5m2_fp16_1(E5_ONE,  E5_ONE,  H_NAN_OUT), H_NAN_OUT);
    CHECK_EQ_HEX(dp32_e5m2_fp16_1(E5_INF,  E5_ONE,  H_ZERO),    H_INF);
    // E5_MAX × E5_MAX = 57344² overflow → +Inf
    CHECK_EQ_HEX(dp32_e5m2_fp16_1(E5_MAX, E5_MAX, H_ZERO), H_INF);
}

static void test_dp32a_e5m2_fp16_chaining() {
    SECTION("DP32A FP16 acc / E5M2 inputs – chaining");

    // 64 × (1×1) = 64
    {
        e5m2_t a[64], bv[64];
        for (int i = 0; i < 64; ++i) { a[i] = E5_ONE; bv[i] = E5_ONE; }
        CHECK_EQ_HEX(hopper_dp32a_e5m2_fp16(a, bv, 64, H_ZERO), h(64.0f));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  DP32A FP32 accumulator – E4M3 inputs
// ─────────────────────────────────────────────────────────────────────────────
static void test_dp32a_e4m3_fp32_basic() {
    SECTION("DP32A FP32 acc / E4M3 inputs – basic");

    CHECK_EQ_HEX(dp32_e4m3_fp32_1(E4_ONE, E4_ONE, F_ZERO),  F_ONE);
    CHECK_EQ_HEX(dp32_e4m3_fp32_1(E4_ONE, E4_ONE, F_ONE),   F_TWO);
    CHECK_EQ_HEX(dp32_e4m3_fp32_1(E4_NEG_ONE, E4_ONE, F_ZERO), F_NEG_ONE);
    CHECK_EQ_HEX(dp32_e4m3_fp32_1(E4_NEG_ZERO, E4_ONE, F_ZERO), F_ZERO);

    // 2×2 + 4 = 8
    {
        e4m3_t a[1]={E4_TWO}, bv[1]={E4_TWO};
        CHECK_EQ_HEX(hopper_dp32a_e4m3_fp32(a, bv, 1, f32(4.0f)), f32(8.0f));
    }

    // 32 × (1×1) + 0 = 32
    {
        e4m3_t a[32], bv[32];
        for (int i = 0; i < 32; ++i) { a[i] = E4_ONE; bv[i] = E4_ONE; }
        CHECK_EQ_HEX(hopper_dp32a_e4m3_fp32(a, bv, 32, F_ZERO), f32(32.0f));
    }
}

static void test_dp32a_e4m3_fp32_specials() {
    SECTION("DP32A FP32 acc / E4M3 inputs – specials");

    // NaN (E4M3: only 0x7F and 0xFF are NaN, i.e. exp=15 AND mant=7)
    CHECK_EQ_HEX(dp32_e4m3_fp32_1(E4_NAN, E4_ONE,  F_ZERO),    F_NAN_OUT);
    CHECK_EQ_HEX(dp32_e4m3_fp32_1(E4_ONE, E4_NAN,  F_ZERO),    F_NAN_OUT);
    // NaN in C
    CHECK_EQ_HEX(dp32_e4m3_fp32_1(E4_ONE, E4_ONE,  F_NAN_OUT), F_NAN_OUT);
    // E4M3 has no Inf → E4_MAX × E4_MAX = 240² = 57600, stays finite
    {
        fp32_t r = dp32_e4m3_fp32_1(E4_MAX, E4_MAX, F_ZERO);
        CHECK(r != F_NAN_OUT && r != F_INF);
    }
    // C = +Inf → result +Inf (no E4M3 product can cancel it)
    CHECK_EQ_HEX(dp32_e4m3_fp32_1(E4_ONE, E4_ONE, F_INF),  F_INF);
}

static void test_dp32a_e4m3_fp32_subnorm() {
    SECTION("DP32A FP32 acc / E4M3 inputs – subnormals");

    {
        fp32_t r = dp32_e4m3_fp32_1(E4_SUBNORM1, E4_ONE, F_ZERO);
        CHECK(r != F_ZERO);
    }
}

static void test_dp32a_e4m3_fp32_chaining() {
    SECTION("DP32A FP32 acc / E4M3 inputs – chaining");

    // 64 × (1×1) = 64
    {
        e4m3_t a[64], bv[64];
        for (int i = 0; i < 64; ++i) { a[i] = E4_ONE; bv[i] = E4_ONE; }
        CHECK_EQ_HEX(hopper_dp32a_e4m3_fp32(a, bv, 64, F_ZERO), f32(64.0f));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  DP32A FP16 accumulator – E4M3 inputs
// ─────────────────────────────────────────────────────────────────────────────
static void test_dp32a_e4m3_fp16_basic() {
    SECTION("DP32A FP16 acc / E4M3 inputs – basic");

    CHECK_EQ_HEX(dp32_e4m3_fp16_1(E4_ONE, E4_ONE, H_ZERO),    H_ONE);
    CHECK_EQ_HEX(dp32_e4m3_fp16_1(E4_ONE, E4_ONE, H_ONE),     H_TWO);
    CHECK_EQ_HEX(dp32_e4m3_fp16_1(E4_NEG_ONE, E4_ONE, H_ZERO), H_NEG_ONE);
    CHECK_EQ_HEX(dp32_e4m3_fp16_1(E4_NEG_ZERO, E4_ONE, H_ZERO), H_ZERO);

    // 32 × (1×1) = 32
    {
        e4m3_t a[32], bv[32];
        for (int i = 0; i < 32; ++i) { a[i] = E4_ONE; bv[i] = E4_ONE; }
        CHECK_EQ_HEX(hopper_dp32a_e4m3_fp16(a, bv, 32, H_ZERO), h(32.0f));
    }
}

static void test_dp32a_e4m3_fp16_specials() {
    SECTION("DP32A FP16 acc / E4M3 inputs – specials");

    CHECK_EQ_HEX(dp32_e4m3_fp16_1(E4_NAN, E4_ONE,  H_ZERO),    H_NAN_OUT);
    CHECK_EQ_HEX(dp32_e4m3_fp16_1(E4_ONE, E4_NAN,  H_ZERO),    H_NAN_OUT);
    CHECK_EQ_HEX(dp32_e4m3_fp16_1(E4_ONE, E4_ONE,  H_NAN_OUT), H_NAN_OUT);
    // C = +Inf passes through (no E4M3 Inf to cancel it)
    CHECK_EQ_HEX(dp32_e4m3_fp16_1(E4_ONE, E4_ONE, H_INF), H_INF);
}

static void test_dp32a_e4m3_fp16_chaining() {
    SECTION("DP32A FP16 acc / E4M3 inputs – chaining");

    // 64 × (1×1) = 64
    {
        e4m3_t a[64], bv[64];
        for (int i = 0; i < 64; ++i) { a[i] = E4_ONE; bv[i] = E4_ONE; }
        CHECK_EQ_HEX(hopper_dp32a_e4m3_fp16(a, bv, 64, H_ZERO), h(64.0f));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::printf("=== Hopper Tensor Core Simulator Tests ===\n");

    test_dp16a_fp32_basic();
    test_dp16a_fp32_specials();
    test_dp16a_fp32_subnorm();
    test_dp16a_fp32_chaining();

    test_dp16a_fp16_basic();
    test_dp16a_fp16_specials();
    test_dp16a_fp16_chaining();

    test_dp16a_bf16_basic();
    test_dp16a_bf16_specials();
    test_dp16a_bf16_chaining();

    test_dp32a_e5m2_fp32_basic();
    test_dp32a_e5m2_fp32_specials();
    test_dp32a_e5m2_fp32_subnorm();
    test_dp32a_e5m2_fp32_chaining();

    test_dp32a_e5m2_fp16_basic();
    test_dp32a_e5m2_fp16_specials();
    test_dp32a_e5m2_fp16_chaining();

    test_dp32a_e4m3_fp32_basic();
    test_dp32a_e4m3_fp32_specials();
    test_dp32a_e4m3_fp32_subnorm();
    test_dp32a_e4m3_fp32_chaining();

    test_dp32a_e4m3_fp16_basic();
    test_dp32a_e4m3_fp16_specials();
    test_dp32a_e4m3_fp16_chaining();

    std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
