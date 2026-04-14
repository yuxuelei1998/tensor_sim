#include "volta_tensor.h"
#include "fp_utils.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

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
#define CHECK_EQ(a,b)       check((a)==(b), #a " == " #b, __FILE__, __LINE__)
#define CHECK_EQ_HEX(a,b)   do { \
    auto _a=(a), _b=(b); \
    if (_a!=_b) { ++g_fail; \
        std::printf("  FAIL  %s:%d  0x%08X != 0x%08X  (%s)\n", \
                    __FILE__,__LINE__,(unsigned)_a,(unsigned)_b, #a " == " #b); \
    } else ++g_pass; } while(0)

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────
static fp16_t h(float f)  { return float_to_fp16(f); }
static fp32_t f32(float f){ return float_to_bits(f); }
static float  tof(fp32_t b){ return bits_to_float(b); }

// FP16 bit constants
static constexpr fp16_t H_ONE      = 0x3C00u;   //  1.0
static constexpr fp16_t H_TWO      = 0x4000u;   //  2.0
static constexpr fp16_t H_NEG_ONE  = 0xBC00u;   // −1.0
static constexpr fp16_t H_HALF     = 0x3800u;   //  0.5
static constexpr fp16_t H_ZERO     = 0x0000u;
static constexpr fp16_t H_NEG_ZERO = 0x8000u;
static constexpr fp16_t H_INF      = 0x7C00u;   //  +∞
static constexpr fp16_t H_NEG_INF  = 0xFC00u;   //  −∞
static constexpr fp16_t H_NAN      = 0x7E00u;   //  qNaN
static constexpr fp16_t H_MAX      = 0x7BFFu;   //  65504  (largest normal FP16)
static constexpr fp16_t H_SUBNORM1 = 0x0001u;   //  2^(−24)  (smallest FP16 subnormal)
static constexpr fp16_t H_SUBNORM2 = 0x0002u;   //  2^(−23)

// FP32 bit constants
static constexpr fp32_t F_ONE     = 0x3F800000u;
static constexpr fp32_t F_TWO     = 0x40000000u;
static constexpr fp32_t F_FOUR    = 0x40800000u;
static constexpr fp32_t F_ZERO    = 0x00000000u;
static constexpr fp32_t F_NEG_ONE = 0xBF800000u;
static constexpr fp32_t F_INF     = 0x7F800000u;
static constexpr fp32_t F_NEG_INF = 0xFF800000u;
static constexpr fp32_t F_NAN_OUT = 0x7FFFFFFFu;
static constexpr fp16_t H_NAN_OUT = 0x7FFFu;

// ─────────────────────────────────────────────────────────────────────────────
//  Section macros
// ─────────────────────────────────────────────────────────────────────────────
#define SECTION(name)  std::printf("\n[%s]\n", name)

// ─────────────────────────────────────────────────────────────────────────────
//  Helper: run DP4A on 1-element vectors (rest zero-padded automatically)
// ─────────────────────────────────────────────────────────────────────────────
static fp32_t dp32_1(fp16_t a0, fp16_t b0, fp32_t c) {
    return volta_dp4a_fp32(&a0, &b0, 1, c);
}
static fp16_t dp16_1(fp16_t a0, fp16_t b0, fp16_t c) {
    return volta_dp4a_fp16(&a0, &b0, 1, c);
}
static fp32_t dp32_4(fp16_t a0,fp16_t a1,fp16_t a2,fp16_t a3,
                     fp16_t b0,fp16_t b1,fp16_t b2,fp16_t b3, fp32_t c) {
    fp16_t a[4]={a0,a1,a2,a3}, b[4]={b0,b1,b2,b3};
    return volta_dp4a_fp32(a,b,4,c);
}
static fp16_t dp16_4(fp16_t a0,fp16_t a1,fp16_t a2,fp16_t a3,
                     fp16_t b0,fp16_t b1,fp16_t b2,fp16_t b3, fp16_t c) {
    fp16_t a[4]={a0,a1,a2,a3}, b[4]={b0,b1,b2,b3};
    return volta_dp4a_fp16(a,b,4,c);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests: FP32 accumulator – basic arithmetic
// ─────────────────────────────────────────────────────────────────────────────
static void test_fp32_basic() {
    SECTION("FP32 accumulator – basic arithmetic");

    // 1×1 + 0 = 1
    CHECK_EQ_HEX(dp32_1(H_ONE, H_ONE, F_ZERO), F_ONE);

    // 4 × (1×1) + 0 = 4
    CHECK_EQ_HEX(dp32_4(H_ONE,H_ONE,H_ONE,H_ONE, H_ONE,H_ONE,H_ONE,H_ONE, F_ZERO), F_FOUR);

    // 1×1 + 1 = 2
    CHECK_EQ_HEX(dp32_1(H_ONE, H_ONE, F_ONE), F_TWO);

    // (−1)×1 + 0 = −1
    CHECK_EQ_HEX(dp32_1(H_NEG_ONE, H_ONE, F_ZERO), F_NEG_ONE);

    // 1×(−1) + 2 = 1
    CHECK_EQ_HEX(dp32_1(H_ONE, H_NEG_ONE, F_TWO), F_ONE);

    // 2×3 + 4 = 10
    {
        fp16_t a[1]={h(2.0f)}, b[1]={h(3.0f)};
        fp32_t res = volta_dp4a_fp32(a,b,1, f32(4.0f));
        CHECK_EQ_HEX(res, f32(10.0f));
    }

    // dot([1,2,3,4],[1,1,1,1]) + 0 = 10
    {
        fp16_t a[4]={h(1),h(2),h(3),h(4)}, b[4]={H_ONE,H_ONE,H_ONE,H_ONE};
        CHECK_EQ_HEX(volta_dp4a_fp32(a,b,4,F_ZERO), f32(10.0f));
    }

    // −0 treated as +0: (−0)×1 + 0 = 0 (not NaN)
    CHECK_EQ_HEX(dp32_1(H_NEG_ZERO, H_ONE, F_ZERO), F_ZERO);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests: FP32 accumulator – special values
// ─────────────────────────────────────────────────────────────────────────────
static void test_fp32_specials() {
    SECTION("FP32 accumulator – special values");

    // NaN in A → NaN output
    CHECK_EQ_HEX(dp32_1(H_NAN, H_ONE, F_ZERO), F_NAN_OUT);
    // NaN in B
    CHECK_EQ_HEX(dp32_1(H_ONE, H_NAN, F_ZERO), F_NAN_OUT);
    // NaN in C
    CHECK_EQ_HEX(dp32_1(H_ONE, H_ONE, 0x7FC00000u), F_NAN_OUT);

    // 0 × ∞ → NaN
    CHECK_EQ_HEX(dp32_1(H_ZERO,    H_INF, F_ZERO), F_NAN_OUT);
    CHECK_EQ_HEX(dp32_1(H_INF,     H_ZERO, F_ZERO), F_NAN_OUT);
    CHECK_EQ_HEX(dp32_1(H_NEG_ZERO,H_INF, F_ZERO), F_NAN_OUT);

    // +∞ + −∞ → NaN  (one product +inf, C = −inf)
    CHECK_EQ_HEX(dp32_1(H_INF, H_ONE, F_NEG_INF), F_NAN_OUT);

    // +∞ × 1 + 0 = +∞
    CHECK_EQ_HEX(dp32_1(H_INF, H_ONE, F_ZERO), F_INF);

    // −∞ × 1 + 0 = −∞
    CHECK_EQ_HEX(dp32_1(H_INF, H_NEG_ONE, F_ZERO), F_NEG_INF);

    // 1×1 with C = +∞  → +∞ (finite + ∞ = ∞)
    CHECK_EQ_HEX(dp32_1(H_ONE, H_ONE, F_INF), F_INF);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests: FP32 accumulator – overflow
// ─────────────────────────────────────────────────────────────────────────────
static void test_fp32_overflow() {
    SECTION("FP32 accumulator – overflow");

    // 65504 × 65504 ≈ 4.29e9, which fits in FP32 (max ~3.4e38)
    {
        fp16_t a[1]={H_MAX}, b[1]={H_MAX};
        fp32_t res = volta_dp4a_fp32(a,b,1,F_ZERO);
        // (65504)^2 = 4290768016 ≈ 2^31.99 → FP32 should represent this (no overflow)
        float v = tof(res);
        CHECK(v > 4.28e9f && v < 4.30e9f);
    }

    // Force FP32 overflow: 4 × (65504)^2 + 0  → overflow to +∞
    {
        fp16_t a[4]={H_MAX,H_MAX,H_MAX,H_MAX}, b[4]={H_MAX,H_MAX,H_MAX,H_MAX};
        fp32_t res = volta_dp4a_fp32(a,b,4,F_ZERO);
        // 4 × 4.29e9 ≈ 1.72e10 still fits in FP32 (< 3.4e38)
        float v = tof(res);
        CHECK(v > 1.7e10f && v < 1.73e10f);
    }

    // FP16 products (max ~4.3e9) are negligible vs fp32_max (~3.4e38):
    // with truncation, fp32_max + tiny = fp32_max (not overflow).
    {
        fp32_t fp32_max = 0x7F7FFFFFu;
        fp32_t res = dp32_1(H_MAX, H_MAX, fp32_max);
        CHECK_EQ_HEX(res, fp32_max);   // truncation keeps fp32_max
    }

    // True FP32 overflow: C just below fp32_max, add FP32 large value as second chain step.
    // Use C = fp32_max and second group adds C_prev as fp32_max → but we chain so
    // the first group output IS fp32_max, fed as C to second group (which has zero products),
    // giving fp32_max again.  Instead test: two groups whose first produces fp32_max as C
    // and second overflows by itself with its own large fp32 C chain.
    // Simplest: use C = FP32 infinity directly (already tested in specials).
    // Verify a large-but-finite case stays finite:
    {
        fp32_t big = 0x4F800000u;  // 2^32 = 4294967296
        fp32_t res = dp32_1(H_ONE, H_ONE, big);
        // 1 + 2^32 = just above 2^32, but truncation might keep 2^32
        // Both fit in FP32, so result should be representable
        CHECK(fp32_exp(res) != 0xFF);  // not infinity or NaN
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests: FP32 accumulator – subnormal FP16 inputs
// ─────────────────────────────────────────────────────────────────────────────
static void test_fp32_subnormal() {
    SECTION("FP32 accumulator – subnormal FP16 inputs");

    // smallest FP16 subnormal (2^-24) × 1 + 0 = 2^-24 in FP32
    {
        fp32_t res = dp32_1(H_SUBNORM1, H_ONE, F_ZERO);
        float  v   = tof(res);
        float  exp = std::ldexpf(1.0f, -24);
        CHECK(std::fabsf(v - exp) < 1e-30f);
    }

    // 2×subnorm1 × 1 = 2^-23
    {
        fp32_t res = dp32_1(H_SUBNORM2, H_ONE, F_ZERO);
        float  v   = tof(res);
        float  exp = std::ldexpf(1.0f, -23);
        CHECK(std::fabsf(v - exp) < 1e-30f);
    }

    // subnorm1 × subnorm1 = 2^-48 (still normal FP32, biased exp = 127-48 = 79)
    {
        fp32_t res = volta_dp4a_fp32(&H_SUBNORM1, &H_SUBNORM1, 1, F_ZERO);
        // 2^-48 in FP32: sign=0, biased_exp=79 (=127-48), mantissa=0
        fp32_t expected = (79u << 23);
        CHECK_EQ_HEX(res, expected);
    }

    // C = smallest FP32 subnormal  (0x00000001) preserved through accumulation
    {
        fp32_t tiny_c = 0x00000001u;   // 2^-149
        fp32_t res = dp32_1(H_ZERO, H_ZERO, tiny_c);  // 0×0 + tiny_c = tiny_c
        CHECK_EQ_HEX(res, tiny_c);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests: FP32 accumulator – long vectors (chained groups)
// ─────────────────────────────────────────────────────────────────────────────
static void test_fp32_long_vectors() {
    SECTION("FP32 accumulator – long vectors");

    // 8 × (1×1) + 0 = 8
    {
        fp16_t a[8], b[8];
        for(int i=0;i<8;++i){a[i]=H_ONE;b[i]=H_ONE;}
        fp32_t res = volta_dp4a_fp32(a,b,8,F_ZERO);
        CHECK_EQ_HEX(res, f32(8.0f));
    }

    // 5 × (1×1) + 0 = 5  (len=5, last group has 1 real + 3 zeros)
    {
        fp16_t a[5], b[5];
        for(int i=0;i<5;++i){a[i]=H_ONE;b[i]=H_ONE;}
        fp32_t res = volta_dp4a_fp32(a,b,5,F_ZERO);
        CHECK_EQ_HEX(res, f32(5.0f));
    }

    // 1 element: result = 1 (vector of length 1)
    {
        fp16_t a[1]={H_ONE}, b[1]={H_ONE};
        CHECK_EQ_HEX(volta_dp4a_fp32(a,b,1,F_ZERO), F_ONE);
    }

    // 12 ones: result = 12
    {
        fp16_t a[12], b[12];
        for(int i=0;i<12;++i){a[i]=H_ONE;b[i]=H_ONE;}
        CHECK_EQ_HEX(volta_dp4a_fp32(a,b,12,F_ZERO), f32(12.0f));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests: FP16 accumulator – basic arithmetic
// ─────────────────────────────────────────────────────────────────────────────
static void test_fp16_basic() {
    SECTION("FP16 accumulator – basic arithmetic");

    // 1×1 + 0 = FP16(1)
    CHECK_EQ_HEX(dp16_1(H_ONE, H_ONE, H_ZERO), H_ONE);

    // 4 × (1×1) + 0 = FP16(4) = 0x4400
    CHECK_EQ_HEX(dp16_4(H_ONE,H_ONE,H_ONE,H_ONE, H_ONE,H_ONE,H_ONE,H_ONE, H_ZERO),
                 (fp16_t)0x4400u);

    // 1×1 + 1 = 2
    CHECK_EQ_HEX(dp16_1(H_ONE, H_ONE, H_ONE), H_TWO);

    // (−1)×1 + 0 = −1
    CHECK_EQ_HEX(dp16_1(H_NEG_ONE, H_ONE, H_ZERO), H_NEG_ONE);

    // 2×3 + 4 = 10  (FP16 10.0 = 0x4900)
    {
        fp16_t a[1]={h(2.0f)}, b[1]={h(3.0f)};
        fp16_t res = volta_dp4a_fp16(a,b,1, h(4.0f));
        CHECK_EQ_HEX(res, h(10.0f));
    }

    // −0 treated as +0: (−0)×1 + 0 = 0
    CHECK_EQ_HEX(dp16_1(H_NEG_ZERO, H_ONE, H_ZERO), H_ZERO);

    // dot([1,2,3,4],[1,1,1,1]) + 0 = 10
    {
        fp16_t a[4]={h(1),h(2),h(3),h(4)}, b[4]={H_ONE,H_ONE,H_ONE,H_ONE};
        CHECK_EQ_HEX(volta_dp4a_fp16(a,b,4,H_ZERO), h(10.0f));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests: FP16 accumulator – special values
// ─────────────────────────────────────────────────────────────────────────────
static void test_fp16_specials() {
    SECTION("FP16 accumulator – special values");

    // NaN
    CHECK_EQ_HEX(dp16_1(H_NAN,  H_ONE,  H_ZERO), H_NAN_OUT);
    CHECK_EQ_HEX(dp16_1(H_ONE,  H_NAN,  H_ZERO), H_NAN_OUT);
    CHECK_EQ_HEX(dp16_1(H_ONE,  H_ONE,  H_NAN),  H_NAN_OUT);

    // 0 × ∞
    CHECK_EQ_HEX(dp16_1(H_ZERO, H_INF,  H_ZERO), H_NAN_OUT);
    CHECK_EQ_HEX(dp16_1(H_INF,  H_ZERO, H_ZERO), H_NAN_OUT);

    // +∞ + −∞
    CHECK_EQ_HEX(dp16_1(H_INF,  H_ONE,  H_NEG_INF), H_NAN_OUT);

    // +∞ × 1 = +∞
    CHECK_EQ_HEX(dp16_1(H_INF, H_ONE, H_ZERO), H_INF);

    // −∞ × 1 = −∞
    CHECK_EQ_HEX(dp16_1(H_INF, H_NEG_ONE, H_ZERO), H_NEG_INF);

    // 1×1 + ∞ = ∞
    CHECK_EQ_HEX(dp16_1(H_ONE, H_ONE, H_INF), H_INF);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests: FP16 accumulator – overflow of product into FP16 range
// ─────────────────────────────────────────────────────────────────────────────
static void test_fp16_overflow() {
    SECTION("FP16 accumulator – overflow");

    // MAX × 2 overflows FP16 → product intermediate is +∞ → result is +∞
    CHECK_EQ_HEX(dp16_1(H_MAX, H_TWO, H_ZERO), H_INF);

    // MAX × MAX = +∞ (intermediate overflows FP16 range)
    CHECK_EQ_HEX(dp16_1(H_MAX, H_MAX, H_ZERO), H_INF);

    // Result overflow via accumulation
    {
        fp16_t a[4]={H_MAX,H_MAX,H_MAX,H_MAX}, b[4]={H_ONE,H_ONE,H_ONE,H_ONE};
        // 4 × 65504 = 262016 > 65504 → overflow
        fp16_t res = volta_dp4a_fp16(a,b,4,H_ZERO);
        CHECK_EQ_HEX(res, H_INF);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests: FP16 accumulator – subnormal inputs
// ─────────────────────────────────────────────────────────────────────────────
static void test_fp16_subnormal() {
    SECTION("FP16 accumulator – subnormal inputs");

    // subnorm1 × 1 + 0 = subnorm1 (value preserved through the pipeline)
    {
        fp16_t res = dp16_1(H_SUBNORM1, H_ONE, H_ZERO);
        CHECK_EQ_HEX(res, H_SUBNORM1);
    }

    // subnorm2 × 1 + 0 = subnorm2
    {
        fp16_t res = dp16_1(H_SUBNORM2, H_ONE, H_ZERO);
        CHECK_EQ_HEX(res, H_SUBNORM2);
    }

    // subnorm1 × subnorm1: product exp = −24+(−24) = −48 << −37 → underflows to 0
    {
        fp16_t res = dp16_1(H_SUBNORM1, H_SUBNORM1, H_ZERO);
        CHECK_EQ_HEX(res, H_ZERO);
    }

    // C = FP16 subnorm (0x0003) + 0 product = 0x0003
    {
        fp16_t sub_c = 0x0003u;
        fp16_t res = dp16_1(H_ZERO, H_ZERO, sub_c);
        CHECK_EQ_HEX(res, sub_c);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests: FP16 accumulator – long vectors
// ─────────────────────────────────────────────────────────────────────────────
static void test_fp16_long_vectors() {
    SECTION("FP16 accumulator – long vectors");

    // 8 ones: 8.0
    {
        fp16_t a[8], b[8];
        for(int i=0;i<8;++i){a[i]=H_ONE;b[i]=H_ONE;}
        CHECK_EQ_HEX(volta_dp4a_fp16(a,b,8,H_ZERO), h(8.0f));
    }

    // 5 ones: 5.0
    {
        fp16_t a[5], b[5];
        for(int i=0;i<5;++i){a[i]=H_ONE;b[i]=H_ONE;}
        CHECK_EQ_HEX(volta_dp4a_fp16(a,b,5,H_ZERO), h(5.0f));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests: FP16 accumulator – RNE rounding
// ─────────────────────────────────────────────────────────────────────────────
static void test_fp16_rne() {
    SECTION("FP16 accumulator – RNE rounding");

    // Construct case where round bit=1, sticky=0 (tie):
    //   result = 1.0 + 0.5×ULP_FP16(1.0) = 1.0 + 2^(-11)
    //   FP16 ULP at 1.0 = 2^(-10).  Half-ULP = 2^(-11).
    //   In 23-bit mantissa space at exp=0: half-ULP = 2^(-11) → mant23 bit 12 set.
    //   mant10 = 0 (even) → round DOWN → result = 1.0
    //
    //   To get a value of 1 + 2^(−11) from the accumulation:
    //     Use C = FP16 1.0 = 0x3C00, product = a×b = 2^(−11)
    //     2^(−11) in FP16: exp_unb = −11, not representable exactly in FP16
    //     (FP16 ULP at 1 is 2^(-10), so 2^(-11) is a sub-ULP value)
    //     Use a FP16 product that sums to give the right intermediate.
    //
    //   Alternative: use two FP16 products summing to give mant23 with bit 12 set.
    //     Product of (0.5)×(2^-10) = 2^-11 can come from:
    //       a = 0.5 = 0x3800,  b = 2^(-10) = FP16 0x1400  (e=5, m=0 → 2^(-10))
    //     a × b = 0.5 × 2^(-10) = 2^(-11) ✓
    //
    //   In fp16_mul_fp16_intermed(0x3800, 0x1400):
    //     na: e=14, exp=−1, sig=1024
    //     nb: e=5,  exp=−10, sig=1024
    //     prod=1024×1024=2^20 (bit-20 case)
    //     mant23 = 0,  result_exp = −1 + (−10) = −11  (normal, since ≥ −14)
    //
    //   With C = 1.0 (exp=0, mant=0):
    //     max_exp = 0.  Contribution from product: shift=0−(−11)=11,
    //       sig=(1<<23)|0=2^23, aligned = 2^23 >> 11 = 2^12 = 4096
    //     Contribution from C: sig=2^23, shift=0, aligned=2^23
    //     accum = 2^23 + 2^12 = 8392704, lead=23
    //     unb = 23+0−23 = 0
    //     mant23 = (8392704 >> 0) & 0x7FFFFF = 8392704 & 0x7FFFFF
    //            = 8392704 mod 8388608 = 4096 = 0x1000
    //     mant10 = 0x1000>>13 = 0,  round_bit=(0x1000>>12)&1=1, sticky=0
    //     mant10 is even (0) → do NOT round up → result = FP16(1.0) = 0x3C00
    {
        fp16_t a_val = 0x3800u;   // 0.5
        fp16_t b_val = 0x1400u;   // 2^(-10)
        fp16_t res = dp16_1(a_val, b_val, H_ONE);
        // 0.5 × 2^(-10) + 1.0 = 1.0 + 2^(-11). Ties to even → 1.0
        CHECK_EQ_HEX(res, H_ONE);
        std::printf("    RNE tie (even mantissa=0): 1.0 + 2^-11 → 0x%04X (expect 0x3C00)\n", res);
    }

    // Same half-ULP but mant10 would be ODD (=1) → round UP
    //   Sum = 1.5×ULP + 1_ULP = 1×ULP + 1.5×ULP ... let me use:
    //   C = 1.0 + 1×ULP = FP16 0x3C01  (mant10=1, odd)
    //   product = same 2^(-11) as above
    //   accum mant23: C's mant = 1<<13 = 0x2000, plus product shifts by 11: 0x1000
    //   mant23 = 0x2000 + 0x1000 = 0x3000
    //   mant10 = 0x3000>>13 = 1 (odd), round_bit=(0x3000>>12)&1=1, sticky=0
    //   mant10 is odd → round UP → mant10=2
    {
        fp16_t c_odd = 0x3C01u;   // FP16 1.0009765625 (1 + 1 ULP)
        fp16_t a_val = 0x3800u;
        fp16_t b_val = 0x1400u;
        fp16_t res = dp16_1(a_val, b_val, c_odd);
        // Expected: mant10 rounded from 1 → 2  → FP16 with mant=2: 0x3C02
        CHECK_EQ_HEX(res, (fp16_t)0x3C02u);
        std::printf("    RNE tie (odd mantissa=1): → 0x%04X (expect 0x3C02)\n", res);
    }

    // Round up (sticky=1): result = 1.0 + 2^(-11) + tiny  → rounds UP to 1 + ULP
    //   Add a second product that makes sticky bit nonzero.
    //   Use second pair with very small product (subnorm1 × subnorm2 = 2^-47 < 2^-37 → 0)
    //   Actually use subnorm1 × H_ONE = 2^-24, which compared to main value 1.0+...
    //   2^-24 >> (0-(-24)) = 2^-24 >> 24 = 0 in integer arithmetic → contributes to sticky
    {
        fp16_t a[4]={0x3800u, H_SUBNORM1, H_ZERO, H_ZERO};
        fp16_t b[4]={0x1400u, H_ONE,      H_ZERO, H_ZERO};
        // Products: 2^(-11) (rounds to zero in accumulation? No: mant23=0, exp=-11, aligned by 11)
        // pd0: exp=-11, aligned shift=11, contribution=2^12
        // pd1: subnorm1 × H_ONE = 2^-24, exp=-24, shift=24 ≥ shift_cap(24) → discarded (no sticky)
        // C=1.0: contribution=2^23
        // accum=2^23+2^12, sticky_total=false (alignment bits discarded per spec)
        // mant23=0x1000, mant10=0, round_bit=1, sticky=FALSE → tie, mant10 even → no round up → 0x3C00
        fp16_t res = volta_dp4a_fp16(a,b,4,H_ONE);
        CHECK_EQ_HEX(res, (fp16_t)0x3C00u);
        std::printf("    RNE sticky=0 (alignment bits discarded): 1.0 + 2^-11 → 0x%04X (expect 0x3C00)\n", res);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests: FP32 accumulator – negative cancellation / sign changes
// ─────────────────────────────────────────────────────────────────────────────
static void test_fp32_cancel() {
    SECTION("FP32 accumulator – cancellation");

    // 1×1 + (−1) = 0
    CHECK_EQ_HEX(dp32_1(H_ONE, H_ONE, F_NEG_ONE), F_ZERO);

    // 2×2 − 4 = 0
    {
        fp16_t a[1]={H_TWO}, b[1]={H_TWO};
        CHECK_EQ_HEX(volta_dp4a_fp32(a,b,1, f32(-4.0f)), F_ZERO);
    }

    // (−1)×(−1) + 0 = 1
    CHECK_EQ_HEX(dp32_1(H_NEG_ONE, H_NEG_ONE, F_ZERO), F_ONE);

    // dot([1,1,1,1],[−1,−1,−1,−1]) + 0 = −4
    {
        fp16_t a[4]={H_ONE,H_ONE,H_ONE,H_ONE};
        fp16_t b[4]={H_NEG_ONE,H_NEG_ONE,H_NEG_ONE,H_NEG_ONE};
        CHECK_EQ_HEX(volta_dp4a_fp32(a,b,4,F_ZERO), f32(-4.0f));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests: FP16 accumulator – negative cancellation
// ─────────────────────────────────────────────────────────────────────────────
static void test_fp16_cancel() {
    SECTION("FP16 accumulator – cancellation");

    // 1×1 + (−1) = 0
    CHECK_EQ_HEX(dp16_1(H_ONE, H_ONE, H_NEG_ONE), H_ZERO);

    // (−1)×(−1) + 0 = 1
    CHECK_EQ_HEX(dp16_1(H_NEG_ONE, H_NEG_ONE, H_ZERO), H_ONE);

    // dot([2,2],[−1,−1]) + 0 = −4  (FP16 −4 = 0xC400)
    {
        fp16_t a[2]={H_TWO,H_TWO}, b[2]={H_NEG_ONE,H_NEG_ONE};
        CHECK_EQ_HEX(volta_dp4a_fp16(a,b,2,H_ZERO), h(-4.0f));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests: cross-check FP32 vs float arithmetic for a range of values
// ─────────────────────────────────────────────────────────────────────────────
static void test_fp32_cross_check() {
    SECTION("FP32 accumulator – cross-check vs float");

    // For simple integer values that are exactly representable, the simulator
    // should match standard float arithmetic (no rounding differences).
    struct TC { float a, b, c; } cases[] = {
        {1,1,0}, {2,3,1}, {-1,4,2}, {0.5f,2,0}, {3,3,3},
        {7,8,9}, {1,1,100}, {16,16,0}, {-5,5,50},
    };
    for (auto& tc : cases) {
        fp16_t a[1]={h(tc.a)}, b[1]={h(tc.b)};
        fp32_t res = volta_dp4a_fp32(a,b,1, f32(tc.c));
        float  got = tof(res);
        float  exp = tc.a * tc.b + tc.c;
        if (std::fabsf(got - exp) > 0.0f) {
            ++g_fail;
            std::printf("  FAIL cross: a=%.1f b=%.1f c=%.1f  got=%.6f exp=%.6f\n",
                        tc.a,tc.b,tc.c,got,exp);
        } else ++g_pass;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::printf("=== Volta Tensor Core Simulator Tests ===\n");

    test_fp32_basic();
    test_fp32_specials();
    test_fp32_overflow();
    test_fp32_subnormal();
    test_fp32_long_vectors();
    test_fp32_cancel();
    test_fp32_cross_check();

    test_fp16_basic();
    test_fp16_specials();
    test_fp16_overflow();
    test_fp16_subnormal();
    test_fp16_long_vectors();
    test_fp16_cancel();
    test_fp16_rne();

    std::printf("\n=========================================\n");
    std::printf("  PASS: %d   FAIL: %d   TOTAL: %d\n", g_pass, g_fail, g_pass+g_fail);
    return g_fail > 0 ? 1 : 0;
}
