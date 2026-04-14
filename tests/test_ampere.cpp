#include "ampere_tensor.h"
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

static constexpr fp16_t H_ONE      = 0x3C00u;
static constexpr fp16_t H_TWO      = 0x4000u;
static constexpr fp16_t H_NEG_ONE  = 0xBC00u;
static constexpr fp16_t H_HALF     = 0x3800u;
static constexpr fp16_t H_ZERO     = 0x0000u;
static constexpr fp16_t H_NEG_ZERO = 0x8000u;
static constexpr fp16_t H_INF      = 0x7C00u;
static constexpr fp16_t H_NEG_INF  = 0xFC00u;
static constexpr fp16_t H_NAN      = 0x7E00u;
static constexpr fp16_t H_MAX      = 0x7BFFu;   // 65504
static constexpr fp16_t H_SUBNORM1 = 0x0001u;   // 2^(−24)

static constexpr fp32_t F_ZERO    = 0x00000000u;
static constexpr fp32_t F_ONE     = 0x3F800000u;
static constexpr fp32_t F_TWO     = 0x40000000u;
static constexpr fp32_t F_FOUR    = 0x40800000u;
static constexpr fp32_t F_EIGHT   = 0x41000000u;
static constexpr fp32_t F_NEG_ONE = 0xBF800000u;
static constexpr fp32_t F_INF     = 0x7F800000u;
static constexpr fp32_t F_NEG_INF = 0xFF800000u;
static constexpr fp32_t F_NAN_OUT = 0x7FFFFFFFu;
static constexpr fp16_t H_NAN_OUT = 0x7FFFu;

// BF16 constants
static constexpr bf16_t BF_ONE     = 0x3F80u;   //  1.0
static constexpr bf16_t BF_TWO     = 0x4000u;   //  2.0
static constexpr bf16_t BF_ZERO    = 0x0000u;
static constexpr bf16_t BF_NEG_ONE = 0xBF80u;   // -1.0
static constexpr bf16_t BF_INF     = 0x7F80u;   // +∞
static constexpr bf16_t BF_NAN     = 0x7FC0u;   //  NaN
static constexpr bf16_t BF_NEG_ZERO= 0x8000u;

// ─────────────────────────────────────────────────────────────────────────────
//  Convenience wrappers: single-element (rest zero-padded by the public API)
// ─────────────────────────────────────────────────────────────────────────────
static fp32_t dp32_1(fp16_t a0, fp16_t b0, fp32_t c) {
    return ampere_dp8a_fp32(&a0, &b0, 1, c);
}
static fp16_t dp16_1(fp16_t a0, fp16_t b0, fp16_t c) {
    return ampere_dp8a_fp16(&a0, &b0, 1, c);
}
static fp32_t dpbf_1(bf16_t a0, bf16_t b0, fp32_t c) {
    return ampere_dp8a_bf16(&a0, &b0, 1, c);
}

// 8-element full-group helpers
static fp32_t dp32_8(fp16_t a0,fp16_t a1,fp16_t a2,fp16_t a3,
                     fp16_t a4,fp16_t a5,fp16_t a6,fp16_t a7,
                     fp16_t b0,fp16_t b1,fp16_t b2,fp16_t b3,
                     fp16_t b4,fp16_t b5,fp16_t b6,fp16_t b7, fp32_t c) {
    fp16_t a[8]={a0,a1,a2,a3,a4,a5,a6,a7};
    fp16_t bv[8]={b0,b1,b2,b3,b4,b5,b6,b7};
    return ampere_dp8a_fp32(a,bv,8,c);
}
static fp16_t dp16_8(fp16_t a0,fp16_t a1,fp16_t a2,fp16_t a3,
                     fp16_t a4,fp16_t a5,fp16_t a6,fp16_t a7,
                     fp16_t b0,fp16_t b1,fp16_t b2,fp16_t b3,
                     fp16_t b4,fp16_t b5,fp16_t b6,fp16_t b7, fp16_t c) {
    fp16_t a[8]={a0,a1,a2,a3,a4,a5,a6,a7};
    fp16_t bv[8]={b0,b1,b2,b3,b4,b5,b6,b7};
    return ampere_dp8a_fp16(a,bv,8,c);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests: FP32 accumulator, FP16 inputs
// ─────────────────────────────────────────────────────────────────────────────
static void test_fp32_basic() {
    SECTION("FP32 accumulator – basic arithmetic (FP16 inputs)");

    // 1×1 + 0 = 1
    CHECK_EQ_HEX(dp32_1(H_ONE, H_ONE, F_ZERO), F_ONE);

    // 8 × (1×1) + 0 = 8
    CHECK_EQ_HEX(dp32_8(H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,
                        H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE, F_ZERO), F_EIGHT);

    // 1×1 + 1 = 2
    CHECK_EQ_HEX(dp32_1(H_ONE, H_ONE, F_ONE), F_TWO);

    // (−1)×1 + 0 = −1
    CHECK_EQ_HEX(dp32_1(H_NEG_ONE, H_ONE, F_ZERO), F_NEG_ONE);

    // 1×(−1) + 2 = 1
    CHECK_EQ_HEX(dp32_1(H_ONE, H_NEG_ONE, F_TWO), F_ONE);

    // 2×3 + 4 = 10
    {
        fp16_t a[1]={h(2.0f)}, bv[1]={h(3.0f)};
        CHECK_EQ_HEX(ampere_dp8a_fp32(a,bv,1,f32(4.0f)), f32(10.0f));
    }

    // dot([1,2,3,4,5,6,7,8],[1,1,1,1,1,1,1,1]) + 0 = 36
    {
        fp16_t a[8]={h(1),h(2),h(3),h(4),h(5),h(6),h(7),h(8)};
        fp16_t bv[8]={H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE};
        CHECK_EQ_HEX(ampere_dp8a_fp32(a,bv,8,F_ZERO), f32(36.0f));
    }

    // −0 treated as +0
    CHECK_EQ_HEX(dp32_1(H_NEG_ZERO, H_ONE, F_ZERO), F_ZERO);
}

static void test_fp32_specials() {
    SECTION("FP32 accumulator – special values");

    // NaN input → NAN_OUT_FP32
    CHECK_EQ_HEX(dp32_1(H_NAN, H_ONE, F_ZERO), F_NAN_OUT);
    CHECK_EQ_HEX(dp32_1(H_ONE, H_NAN, F_ZERO), F_NAN_OUT);

    // 0 × ∞ → NAN_OUT_FP32
    CHECK_EQ_HEX(dp32_1(H_ZERO, H_INF, F_ZERO), F_NAN_OUT);
    CHECK_EQ_HEX(dp32_1(H_INF,  H_ZERO, F_ZERO), F_NAN_OUT);

    // +∞ + −∞ → NAN_OUT_FP32
    CHECK_EQ_HEX(dp32_1(H_INF, H_ONE,     F_NEG_INF), F_NAN_OUT);
    CHECK_EQ_HEX(dp32_1(H_INF, H_NEG_ONE, F_INF),     F_NAN_OUT);

    // NaN in C → NAN_OUT_FP32
    CHECK_EQ_HEX(dp32_1(H_ONE, H_ONE, F_NAN_OUT), F_NAN_OUT);

    // ∞ × 1 → +∞
    CHECK_EQ_HEX(dp32_1(H_INF, H_ONE, F_ZERO), F_INF);

    // ∞ × (−1) → −∞
    CHECK_EQ_HEX(dp32_1(H_INF, H_NEG_ONE, F_ZERO), F_NEG_INF);

    // FP16 inputs cannot overflow FP32 accumulator:
    // 65504 × 65504 ≈ 4.29e9, unbiased exp=31, biased FP32 exp=158 < 255 → finite
    // Just verify it doesn't return NaN or Inf
    {
        fp32_t r = dp32_1(H_MAX, H_MAX, F_ZERO);
        CHECK(r != F_NAN_OUT && r != F_INF && r != F_NEG_INF);
    }
}

static void test_fp32_subnorm() {
    SECTION("FP32 accumulator – subnormals");

    // subnormal × 1 + 0: result should be non-zero
    {
        fp32_t r = dp32_1(H_SUBNORM1, H_ONE, F_ZERO);
        CHECK(r != F_ZERO);
    }

    // subnormal × subnormal: H_SUBNORM1 = 2^(-24), product = 2^(-48)
    // In FP32 this is a normal value (biased exp = 79), not zero
    CHECK_EQ_HEX(dp32_1(H_SUBNORM1, H_SUBNORM1, F_ZERO), (fp32_t)0x27800000u);
}

static void test_fp32_chaining() {
    SECTION("FP32 accumulator – multi-group chaining");

    // 16 × (1×1) should equal 16 (two groups of 8)
    {
        fp16_t a[16], bv[16];
        for (int i = 0; i < 16; ++i) { a[i] = H_ONE; bv[i] = H_ONE; }
        CHECK_EQ_HEX(ampere_dp8a_fp32(a,bv,16,F_ZERO), f32(16.0f));
    }

    // Vector length 9 (one full group of 8 + partial of 1)
    {
        fp16_t a[9], bv[9];
        for (int i = 0; i < 9; ++i) { a[i] = H_ONE; bv[i] = H_ONE; }
        CHECK_EQ_HEX(ampere_dp8a_fp32(a,bv,9,F_ZERO), f32(9.0f));
    }

    // Length 7 (partial group only): dot([1..7],[1..7]) = 1+4+9+16+25+36+49 = 140
    {
        fp16_t a[7], bv[7];
        for (int i = 0; i < 7; ++i) { a[i] = h((float)(i+1)); bv[i] = h((float)(i+1)); }
        CHECK_EQ_HEX(ampere_dp8a_fp32(a,bv,7,F_ZERO), f32(140.0f));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests: FP16 accumulator, FP16 inputs
// ─────────────────────────────────────────────────────────────────────────────
static void test_fp16_basic() {
    SECTION("FP16 accumulator – basic arithmetic");

    // 1×1 + 0 = 1
    CHECK_EQ_HEX(dp16_1(H_ONE, H_ONE, H_ZERO), H_ONE);

    // 8 × (1×1) + 0 = 8
    CHECK_EQ_HEX(dp16_8(H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,
                        H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE, H_ZERO),
                 h(8.0f));

    // 1×1 + 1 = 2
    CHECK_EQ_HEX(dp16_1(H_ONE, H_ONE, H_ONE), H_TWO);

    // (−1)×1 + 0 = −1
    CHECK_EQ_HEX(dp16_1(H_NEG_ONE, H_ONE, H_ZERO), H_NEG_ONE);

    // −0 treated as +0
    CHECK_EQ_HEX(dp16_1(H_NEG_ZERO, H_ONE, H_ZERO), H_ZERO);
}

static void test_fp16_specials() {
    SECTION("FP16 accumulator – special values");

    // NaN input → NAN_OUT_FP16
    CHECK_EQ_HEX(dp16_1(H_NAN, H_ONE, H_ZERO), H_NAN_OUT);
    CHECK_EQ_HEX(dp16_1(H_ONE, H_NAN, H_ZERO), H_NAN_OUT);

    // 0 × ∞ → NAN_OUT_FP16
    CHECK_EQ_HEX(dp16_1(H_ZERO, H_INF, H_ZERO), H_NAN_OUT);

    // +∞ + −∞ → NAN_OUT_FP16
    CHECK_EQ_HEX(dp16_1(H_INF, H_ONE, H_NEG_INF), H_NAN_OUT);

    // NaN in C → NAN_OUT_FP16
    CHECK_EQ_HEX(dp16_1(H_ONE, H_ONE, H_NAN_OUT), H_NAN_OUT);

    // ∞ × 1 = +∞
    CHECK_EQ_HEX(dp16_1(H_INF, H_ONE, H_ZERO), H_INF);

    // overflow → ±∞
    CHECK_EQ_HEX(dp16_1(H_MAX, H_MAX, H_ZERO), H_INF);
}

static void test_fp16_rne() {
    SECTION("FP16 accumulator – RNE rounding");

    // 0.5 × 0.5 + 0 = 0.25  (exact, no rounding needed)
    CHECK_EQ_HEX(dp16_1(H_HALF, H_HALF, H_ZERO), h(0.25f));

    // dot([1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1]) + 0 = 8  (exact)
    CHECK_EQ_HEX(dp16_8(H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,
                        H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE, H_ZERO),
                 h(8.0f));
}

static void test_fp16_chaining() {
    SECTION("FP16 accumulator – multi-group chaining");

    // 16 × (1×1) = 16
    {
        fp16_t a[16], bv[16];
        for (int i = 0; i < 16; ++i) { a[i] = H_ONE; bv[i] = H_ONE; }
        CHECK_EQ_HEX(ampere_dp8a_fp16(a,bv,16,H_ZERO), h(16.0f));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests: FP32 accumulator, BF16 inputs
// ─────────────────────────────────────────────────────────────────────────────
static void test_bf16_basic() {
    SECTION("BF16 inputs, FP32 accumulator – basic arithmetic");

    // 1×1 + 0 = 1
    CHECK_EQ_HEX(dpbf_1(BF_ONE, BF_ONE, F_ZERO), F_ONE);

    // 1×1 + 1 = 2
    CHECK_EQ_HEX(dpbf_1(BF_ONE, BF_ONE, F_ONE), F_TWO);

    // (−1)×1 + 0 = −1
    CHECK_EQ_HEX(dpbf_1(BF_NEG_ONE, BF_ONE, F_ZERO), F_NEG_ONE);

    // 2×3 + 4 = 10
    {
        bf16_t a[1]={b(2.0f)}, bv[1]={b(3.0f)};
        CHECK_EQ_HEX(ampere_dp8a_bf16(a,bv,1,f32(4.0f)), f32(10.0f));
    }

    // 8 × (1×1) + 0 = 8
    {
        bf16_t a[8], bv[8];
        for (int i = 0; i < 8; ++i) { a[i] = BF_ONE; bv[i] = BF_ONE; }
        CHECK_EQ_HEX(ampere_dp8a_bf16(a,bv,8,F_ZERO), f32(8.0f));
    }

    // −0 treated as +0
    CHECK_EQ_HEX(dpbf_1(BF_NEG_ZERO, BF_ONE, F_ZERO), F_ZERO);
}

static void test_bf16_specials() {
    SECTION("BF16 inputs, FP32 accumulator – special values");

    // NaN → NAN_OUT_FP32
    CHECK_EQ_HEX(dpbf_1(BF_NAN, BF_ONE, F_ZERO), F_NAN_OUT);

    // 0 × ∞ → NAN_OUT_FP32
    CHECK_EQ_HEX(dpbf_1(BF_ZERO, BF_INF, F_ZERO), F_NAN_OUT);

    // +∞ + −∞ → NAN_OUT_FP32
    CHECK_EQ_HEX(dpbf_1(BF_INF, BF_ONE, F_NEG_INF), F_NAN_OUT);

    // ∞ × 1 = +∞
    CHECK_EQ_HEX(dpbf_1(BF_INF, BF_ONE, F_ZERO), F_INF);
}

static void test_bf16_chaining() {
    SECTION("BF16 inputs, FP32 accumulator – multi-group chaining");

    // 16 × (1×1) = 16
    {
        bf16_t a[16], bv[16];
        for (int i = 0; i < 16; ++i) { a[i] = BF_ONE; bv[i] = BF_ONE; }
        CHECK_EQ_HEX(ampere_dp8a_bf16(a,bv,16,F_ZERO), f32(16.0f));
    }

    // Length 5 (partial group)
    {
        bf16_t a[5], bv[5];
        for (int i = 0; i < 5; ++i) { a[i] = BF_TWO; bv[i] = BF_TWO; }
        // 5 × (2×2) = 20
        CHECK_EQ_HEX(ampere_dp8a_bf16(a,bv,5,F_ZERO), f32(20.0f));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::printf("=== Ampere DP8A Simulator Tests ===\n");

    test_fp32_basic();
    test_fp32_specials();
    test_fp32_subnorm();
    test_fp32_chaining();

    test_fp16_basic();
    test_fp16_specials();
    test_fp16_rne();
    test_fp16_chaining();

    test_bf16_basic();
    test_bf16_specials();
    test_bf16_chaining();

    std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
