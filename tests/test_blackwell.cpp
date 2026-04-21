#include "blackwell_tensor.h"
#include "fp_utils.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

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

static fp16_t h(float f)   { return float_to_fp16(f); }
static bf16_t b(float f)   { return float_to_bf16(f); }
static fp32_t f32(float f) { return float_to_bits(f); }

static constexpr fp16_t H_ONE      = 0x3C00u;
static constexpr fp16_t H_TWO      = 0x4000u;
static constexpr fp16_t H_NEG_ONE  = 0xBC00u;
static constexpr fp16_t H_ZERO     = 0x0000u;
static constexpr fp16_t H_NEG_ZERO = 0x8000u;
static constexpr fp16_t H_INF      = 0x7C00u;
static constexpr fp16_t H_NEG_INF  = 0xFC00u;
static constexpr fp16_t H_NAN      = 0x7E00u;
static constexpr fp16_t H_MAX      = 0x7BFFu;
static constexpr fp16_t H_SUBNORM1 = 0x0001u;

static constexpr fp32_t F_ZERO    = 0x00000000u;
static constexpr fp32_t F_ONE     = 0x3F800000u;
static constexpr fp32_t F_TWO     = 0x40000000u;
static constexpr fp32_t F_NEG_ONE = 0xBF800000u;
static constexpr fp32_t F_INF     = 0x7F800000u;
static constexpr fp32_t F_NEG_INF = 0xFF800000u;
static constexpr fp32_t F_NAN_OUT = 0x7FFFFFFFu;
static constexpr fp16_t H_NAN_OUT = 0x7FFFu;

static constexpr bf16_t BF_ONE     = 0x3F80u;
static constexpr bf16_t BF_TWO     = 0x4000u;
static constexpr bf16_t BF_ZERO    = 0x0000u;
static constexpr bf16_t BF_NEG_ONE = 0xBF80u;
static constexpr bf16_t BF_INF     = 0x7F80u;
static constexpr bf16_t BF_NAN     = 0x7FC0u;

static constexpr e5m2_t E5_ONE      = 0x3Cu;
static constexpr e5m2_t E5_TWO      = 0x40u;
static constexpr e5m2_t E5_ZERO     = 0x00u;
static constexpr e5m2_t E5_NEG_ONE  = 0xBCu;
static constexpr e5m2_t E5_NEG_ZERO = 0x80u;
static constexpr e5m2_t E5_INF      = 0x7Cu;
static constexpr e5m2_t E5_NEG_INF  = 0xFCu;
static constexpr e5m2_t E5_NAN      = 0x7Du;
static constexpr e5m2_t E5_SUBNORM1 = 0x01u;
static constexpr e5m2_t E5_MAX      = 0x7Bu;

static constexpr e4m3_t E4_ONE      = 0x38u;
static constexpr e4m3_t E4_TWO      = 0x40u;
static constexpr e4m3_t E4_ZERO     = 0x00u;
static constexpr e4m3_t E4_NEG_ZERO = 0x80u;
static constexpr e4m3_t E4_NEG_ONE  = 0xB8u;
static constexpr e4m3_t E4_NAN      = 0x7Fu;
static constexpr e4m3_t E4_SUBNORM1 = 0x01u;
static constexpr e4m3_t E4_MAX      = 0x77u;
static constexpr e4m3_t E4_MAX_TRUE = 0x7Eu;
static fp32_t dp16_fp32_1(fp16_t a0, fp16_t b0, fp32_t c) {
    return blackwell_dp16a_fp32(&a0, &b0, 1, c);
}
static fp16_t dp16_fp16_1(fp16_t a0, fp16_t b0, fp16_t c) {
    return blackwell_dp16a_fp16(&a0, &b0, 1, c);
}
static fp32_t dp16_bf16_1(bf16_t a0, bf16_t b0, fp32_t c) {
    return blackwell_dp16a_bf16(&a0, &b0, 1, c);
}
static fp32_t dp32_e5m2_fp32_1(e5m2_t a0, e5m2_t b0, fp32_t c) {
    return blackwell_dp32a_e5m2_fp32(&a0, &b0, 1, c);
}
static fp16_t dp32_e5m2_fp16_1(e5m2_t a0, e5m2_t b0, fp16_t c) {
    return blackwell_dp32a_e5m2_fp16(&a0, &b0, 1, c);
}
static fp32_t dp32_e4m3_fp32_1(e4m3_t a0, e4m3_t b0, fp32_t c) {
    return blackwell_dp32a_e4m3_fp32(&a0, &b0, 1, c);
}
static fp16_t dp32_e4m3_fp16_1(e4m3_t a0, e4m3_t b0, fp16_t c) {
    return blackwell_dp32a_e4m3_fp16(&a0, &b0, 1, c);
}

static void test_dp16a_fp32_basic() {
    SECTION("DP16A FP32 acc / FP16 inputs – basic");
    CHECK_EQ_HEX(dp16_fp32_1(H_ONE, H_ONE, F_ZERO),  F_ONE);
    CHECK_EQ_HEX(dp16_fp32_1(H_ONE, H_ONE, F_ONE),   F_TWO);
    CHECK_EQ_HEX(dp16_fp32_1(H_NEG_ONE, H_ONE, F_ZERO), F_NEG_ONE);
    CHECK_EQ_HEX(dp16_fp32_1(H_NEG_ZERO, H_ONE, F_ZERO), F_ZERO);
    {
        fp16_t a[1]={h(2.0f)}, bv[1]={h(3.0f)};
        CHECK_EQ_HEX(blackwell_dp16a_fp32(a, bv, 1, f32(4.0f)), f32(10.0f));
    }
    {
        fp16_t a[16], bv[16];
        for (int i = 0; i < 16; ++i) { a[i] = H_ONE; bv[i] = H_ONE; }
        CHECK_EQ_HEX(blackwell_dp16a_fp32(a, bv, 16, F_ZERO), f32(16.0f));
    }
}

static void test_dp16a_fp32_specials() {
    SECTION("DP16A FP32 acc / FP16 inputs – specials");
    CHECK_EQ_HEX(dp16_fp32_1(H_NAN,  H_ONE,  F_ZERO),    F_NAN_OUT);
    CHECK_EQ_HEX(dp16_fp32_1(H_ZERO, H_INF,  F_ZERO),    F_NAN_OUT);
    CHECK_EQ_HEX(dp16_fp32_1(H_INF,  H_ONE,  F_NEG_INF), F_NAN_OUT);
    CHECK_EQ_HEX(dp16_fp32_1(H_ONE,  H_ONE,  F_NAN_OUT), F_NAN_OUT);
    CHECK_EQ_HEX(dp16_fp32_1(H_INF,  H_ONE,  F_ZERO),    F_INF);
    CHECK_EQ_HEX(dp16_fp32_1(H_INF,  H_NEG_ONE, F_ZERO), F_NEG_INF);
}

static void test_dp16a_bf16_basic() {
    SECTION("DP16A FP32 acc / BF16 inputs – basic");
    CHECK_EQ_HEX(dp16_bf16_1(BF_ONE, BF_ONE, F_ZERO), F_ONE);
    CHECK_EQ_HEX(dp16_bf16_1(BF_ONE, BF_ONE, F_ONE),  F_TWO);
    CHECK_EQ_HEX(dp16_bf16_1(BF_NEG_ONE, BF_ONE, F_ZERO), F_NEG_ONE);
    {
        bf16_t a[1]={b(2.0f)}, bv[1]={b(3.0f)};
        CHECK_EQ_HEX(blackwell_dp16a_bf16(a, bv, 1, f32(4.0f)), f32(10.0f));
    }
}

static void test_dp32a_e5m2_fp32_basic() {
    SECTION("DP32A FP32 acc / E5M2 inputs – basic");
    CHECK_EQ_HEX(dp32_e5m2_fp32_1(E5_ONE, E5_ONE, F_ZERO),  F_ONE);
    CHECK_EQ_HEX(dp32_e5m2_fp32_1(E5_ONE, E5_ONE, F_ONE),   F_TWO);
    CHECK_EQ_HEX(dp32_e5m2_fp32_1(E5_NEG_ONE, E5_ONE, F_ZERO), F_NEG_ONE);
    CHECK_EQ_HEX(dp32_e5m2_fp32_1(E5_NEG_ZERO, E5_ONE, F_ZERO), F_ZERO);
    {
        e5m2_t a[1]={E5_TWO}, bv[1]={E5_TWO};
        CHECK_EQ_HEX(blackwell_dp32a_e5m2_fp32(a, bv, 1, f32(4.0f)), f32(8.0f));
    }
    {
        e5m2_t a[32], bv[32];
        for (int i = 0; i < 32; ++i) { a[i] = E5_ONE; bv[i] = E5_ONE; }
        CHECK_EQ_HEX(blackwell_dp32a_e5m2_fp32(a, bv, 32, F_ZERO), f32(32.0f));
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
    {
        fp32_t r = dp32_e5m2_fp32_1(E5_SUBNORM1, E5_ONE, F_ZERO);
        CHECK(r != F_ZERO);
    }
    {
        fp32_t r = dp32_e5m2_fp32_1(E5_SUBNORM1, E5_SUBNORM1, F_ZERO);
        CHECK(r != F_ZERO);
    }
}

static void test_dp32a_e5m2_fp32_precision() {
    SECTION("DP32A FP32 acc / E5M2 inputs – 25-bit precision");
    {
        e5m2_t a = 0x3Du, bv = 0x3Eu;
        fp32_t r = dp32_e5m2_fp32_1(a, bv, F_ZERO);
        CHECK_EQ_HEX(r, f32(1.875f));
    }
    {
        fp32_t r = dp32_e5m2_fp32_1(E5_MAX, E5_MAX, F_ZERO);
        CHECK(r != F_NAN_OUT && r != F_INF);
    }
}

static void test_dp32a_e5m2_fp32_chaining() {
    SECTION("DP32A FP32 acc / E5M2 inputs – chaining");
    {
        e5m2_t a[64], bv[64];
        for (int i = 0; i < 64; ++i) { a[i] = E5_ONE; bv[i] = E5_ONE; }
        CHECK_EQ_HEX(blackwell_dp32a_e5m2_fp32(a, bv, 64, F_ZERO), f32(64.0f));
    }
    {
        e5m2_t a[33], bv[33];
        for (int i = 0; i < 33; ++i) { a[i] = E5_ONE; bv[i] = E5_ONE; }
        CHECK_EQ_HEX(blackwell_dp32a_e5m2_fp32(a, bv, 33, F_ZERO), f32(33.0f));
    }
}

static void test_dp32a_e5m2_fp16_basic() {
    SECTION("DP32A FP16 acc / E5M2 inputs – basic");
    CHECK_EQ_HEX(dp32_e5m2_fp16_1(E5_ONE, E5_ONE, H_ZERO),    H_ONE);
    CHECK_EQ_HEX(dp32_e5m2_fp16_1(E5_ONE, E5_ONE, H_ONE),     H_TWO);
    CHECK_EQ_HEX(dp32_e5m2_fp16_1(E5_NEG_ONE, E5_ONE, H_ZERO), H_NEG_ONE);
    CHECK_EQ_HEX(dp32_e5m2_fp16_1(E5_NEG_ZERO, E5_ONE, H_ZERO), H_ZERO);
    {
        e5m2_t a[32], bv[32];
        for (int i = 0; i < 32; ++i) { a[i] = E5_ONE; bv[i] = E5_ONE; }
        CHECK_EQ_HEX(blackwell_dp32a_e5m2_fp16(a, bv, 32, H_ZERO), h(32.0f));
    }
}

static void test_dp32a_e5m2_fp16_specials() {
    SECTION("DP32A FP16 acc / E5M2 inputs – specials");
    CHECK_EQ_HEX(dp32_e5m2_fp16_1(E5_NAN,  E5_ONE,  H_ZERO),    H_NAN_OUT);
    CHECK_EQ_HEX(dp32_e5m2_fp16_1(E5_ZERO, E5_INF,  H_ZERO),    H_NAN_OUT);
    CHECK_EQ_HEX(dp32_e5m2_fp16_1(E5_INF,  E5_ONE,  H_NEG_INF), H_NAN_OUT);
    CHECK_EQ_HEX(dp32_e5m2_fp16_1(E5_ONE,  E5_ONE,  H_NAN_OUT), H_NAN_OUT);
    CHECK_EQ_HEX(dp32_e5m2_fp16_1(E5_INF,  E5_ONE,  H_ZERO),    H_INF);
}

static void test_dp32a_e4m3_fp32_basic() {
    SECTION("DP32A FP32 acc / E4M3 inputs – basic");
    CHECK_EQ_HEX(dp32_e4m3_fp32_1(E4_ONE, E4_ONE, F_ZERO),  F_ONE);
    CHECK_EQ_HEX(dp32_e4m3_fp32_1(E4_ONE, E4_ONE, F_ONE),   F_TWO);
    CHECK_EQ_HEX(dp32_e4m3_fp32_1(E4_NEG_ONE, E4_ONE, F_ZERO), F_NEG_ONE);
    CHECK_EQ_HEX(dp32_e4m3_fp32_1(E4_NEG_ZERO, E4_ONE, F_ZERO), F_ZERO);
    {
        e4m3_t a[1]={E4_TWO}, bv[1]={E4_TWO};
        CHECK_EQ_HEX(blackwell_dp32a_e4m3_fp32(a, bv, 1, f32(4.0f)), f32(8.0f));
    }
    {
        e4m3_t a[32], bv[32];
        for (int i = 0; i < 32; ++i) { a[i] = E4_ONE; bv[i] = E4_ONE; }
        CHECK_EQ_HEX(blackwell_dp32a_e4m3_fp32(a, bv, 32, F_ZERO), f32(32.0f));
    }
}

static void test_dp32a_e4m3_fp32_specials() {
    SECTION("DP32A FP32 acc / E4M3 inputs – specials");
    CHECK_EQ_HEX(dp32_e4m3_fp32_1(E4_NAN, E4_ONE,  F_ZERO),    F_NAN_OUT);
    CHECK_EQ_HEX(dp32_e4m3_fp32_1(E4_ONE, E4_NAN,  F_ZERO),    F_NAN_OUT);
    CHECK_EQ_HEX(dp32_e4m3_fp32_1(E4_ONE, E4_ONE,  F_NAN_OUT), F_NAN_OUT);
    {
        fp32_t r = dp32_e4m3_fp32_1(E4_MAX, E4_MAX, F_ZERO);
        CHECK(r != F_NAN_OUT && r != F_INF);
    }
    {
        fp32_t r = dp32_e4m3_fp32_1(E4_MAX_TRUE, E4_MAX_TRUE, F_ZERO);
        CHECK(r != F_NAN_OUT && r != F_INF);
    }
    CHECK_EQ_HEX(dp32_e4m3_fp32_1(E4_ONE, E4_ONE, F_INF),  F_INF);
}

static void test_dp32a_e4m3_fp32_precision() {
    SECTION("DP32A FP32 acc / E4M3 inputs – 25-bit precision");
    {
        e4m3_t a = 0x3Au, bv = 0x3Cu;
        fp32_t r = dp32_e4m3_fp32_1(a, bv, F_ZERO);
        CHECK_EQ_HEX(r, f32(1.875f));
    }
    {
        e4m3_t a = 0x3Fu, bv = 0x3Fu;
        fp32_t r = dp32_e4m3_fp32_1(a, bv, F_ZERO);
        CHECK_EQ_HEX(r, f32(3.515625f));
    }
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
    {
        e4m3_t a[64], bv[64];
        for (int i = 0; i < 64; ++i) { a[i] = E4_ONE; bv[i] = E4_ONE; }
        CHECK_EQ_HEX(blackwell_dp32a_e4m3_fp32(a, bv, 64, F_ZERO), f32(64.0f));
    }
}

static void test_dp32a_e4m3_fp16_basic() {
    SECTION("DP32A FP16 acc / E4M3 inputs – basic");
    CHECK_EQ_HEX(dp32_e4m3_fp16_1(E4_ONE, E4_ONE, H_ZERO),    H_ONE);
    CHECK_EQ_HEX(dp32_e4m3_fp16_1(E4_ONE, E4_ONE, H_ONE),     H_TWO);
    CHECK_EQ_HEX(dp32_e4m3_fp16_1(E4_NEG_ONE, E4_ONE, H_ZERO), H_NEG_ONE);
    CHECK_EQ_HEX(dp32_e4m3_fp16_1(E4_NEG_ZERO, E4_ONE, H_ZERO), H_ZERO);
    {
        e4m3_t a[32], bv[32];
        for (int i = 0; i < 32; ++i) { a[i] = E4_ONE; bv[i] = E4_ONE; }
        CHECK_EQ_HEX(blackwell_dp32a_e4m3_fp16(a, bv, 32, H_ZERO), h(32.0f));
    }
}

static void test_dp32a_e4m3_fp16_specials() {
    SECTION("DP32A FP16 acc / E4M3 inputs – specials");
    CHECK_EQ_HEX(dp32_e4m3_fp16_1(E4_NAN, E4_ONE,  H_ZERO),    H_NAN_OUT);
    CHECK_EQ_HEX(dp32_e4m3_fp16_1(E4_ONE, E4_NAN,  H_ZERO),    H_NAN_OUT);
    CHECK_EQ_HEX(dp32_e4m3_fp16_1(E4_ONE, E4_ONE,  H_NAN_OUT), H_NAN_OUT);
    CHECK_EQ_HEX(dp32_e4m3_fp16_1(E4_ONE, E4_ONE,  H_INF),     H_INF);
}

static void test_dp32a_e4m3_fp16_chaining() {
    SECTION("DP32A FP16 acc / E4M3 inputs – chaining");
    {
        e4m3_t a[64], bv[64];
        for (int i = 0; i < 64; ++i) { a[i] = E4_ONE; bv[i] = E4_ONE; }
        CHECK_EQ_HEX(blackwell_dp32a_e4m3_fp16(a, bv, 64, H_ZERO), h(64.0f));
    }
}

static void test_dp32a_e2m1_fp32_basic() {
    SECTION("DP32A FP32 acc / E2M1 inputs – basic");
    {
        e2m1_t a[64] = {0}, bv[64] = {0};
        a[0] = 0x2; bv[0] = 0x2;
        CHECK_EQ_HEX(blackwell_dp32a_e2m1_fp32(a, bv, 64, F_ZERO), f32(1.0f));
    }
    {
        e2m1_t a[64], bv[64];
        for (int i = 0; i < 64; ++i) { a[i] = 0x2; bv[i] = 0x2; }
        CHECK_EQ_HEX(blackwell_dp32a_e2m1_fp32(a, bv, 64, F_ZERO), f32(64.0f));
    }
    {
        e2m1_t a[64], bv[64];
        for (int i = 0; i < 64; ++i) { a[i] = 0x3; bv[i] = 0x3; }
        CHECK_EQ_HEX(blackwell_dp32a_e2m1_fp32(a, bv, 64, F_ZERO), f32(144.0f));
    }
}

static void test_dp32a_e2m1_fp32_subnorm() {
    SECTION("DP32A FP32 acc / E2M1 inputs – subnormals");
    {
        e2m1_t a[64] = {0}, bv[64] = {0};
        a[0] = 0x1; bv[0] = 0x2;
        CHECK_EQ_HEX(blackwell_dp32a_e2m1_fp32(a, bv, 64, F_ZERO), f32(0.5f));
    }
    {
        e2m1_t a[64] = {0}, bv[64] = {0};
        a[0] = 0x1; bv[0] = 0x1;
        CHECK_EQ_HEX(blackwell_dp32a_e2m1_fp32(a, bv, 64, F_ZERO), f32(0.25f));
    }
}

static void test_dp32a_e2m1_fp32_chaining() {
    SECTION("DP32A FP32 acc / E2M1 inputs – chaining");
    {
        e2m1_t a[65], bv[65];
        for (int i = 0; i < 65; ++i) { a[i] = 0x2; bv[i] = 0x2; }
        CHECK_EQ_HEX(blackwell_dp32a_e2m1_fp32(a, bv, 65, F_ZERO), f32(65.0f));
    }
}

static void test_dp32a_e2m1_fp16_basic() {
    SECTION("DP32A FP16 acc / E2M1 inputs – basic");
    {
        e2m1_t a[64] = {0}, bv[64] = {0};
        a[0] = 0x2; bv[0] = 0x2;
        CHECK_EQ_HEX(blackwell_dp32a_e2m1_fp16(a, bv, 64, H_ZERO), h(1.0f));
    }
    {
        e2m1_t a[64], bv[64];
        for (int i = 0; i < 64; ++i) { a[i] = 0x2; bv[i] = 0x2; }
        CHECK_EQ_HEX(blackwell_dp32a_e2m1_fp16(a, bv, 64, H_ZERO), h(64.0f));
    }
}

static void test_dp32a_e2m1_fp16_chaining() {
    SECTION("DP32A FP16 acc / E2M1 inputs – chaining");
    {
        e2m1_t a[65], bv[65];
        for (int i = 0; i < 65; ++i) { a[i] = 0x2; bv[i] = 0x2; }
        CHECK_EQ_HEX(blackwell_dp32a_e2m1_fp16(a, bv, 65, H_ZERO), h(65.0f));
    }
}

int main() {
    std::printf("=== Blackwell Tensor Core Simulator Tests ===\n");

    test_dp16a_fp32_basic();
    test_dp16a_fp32_specials();
    test_dp16a_bf16_basic();

    test_dp32a_e5m2_fp32_basic();
    test_dp32a_e5m2_fp32_specials();
    test_dp32a_e5m2_fp32_subnorm();
    test_dp32a_e5m2_fp32_precision();
    test_dp32a_e5m2_fp32_chaining();
    test_dp32a_e5m2_fp16_basic();
    test_dp32a_e5m2_fp16_specials();

    test_dp32a_e4m3_fp32_basic();
    test_dp32a_e4m3_fp32_specials();
    test_dp32a_e4m3_fp32_precision();
    test_dp32a_e4m3_fp32_subnorm();
    test_dp32a_e4m3_fp32_chaining();
    test_dp32a_e4m3_fp16_basic();
    test_dp32a_e4m3_fp16_specials();
    test_dp32a_e4m3_fp16_chaining();

    test_dp32a_e2m1_fp32_basic();
    test_dp32a_e2m1_fp32_subnorm();
    test_dp32a_e2m1_fp32_chaining();
    test_dp32a_e2m1_fp16_basic();
    test_dp32a_e2m1_fp16_chaining();

    std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}