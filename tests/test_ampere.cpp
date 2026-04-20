#include "ampere_tensor.h"
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
static constexpr fp16_t H_HALF     = 0x3800u;
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
static constexpr fp32_t F_FOUR    = 0x40800000u;
static constexpr fp32_t F_EIGHT   = 0x41000000u;
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
static constexpr bf16_t BF_NEG_ZERO= 0x8000u;

static fp32_t dp32_1(fp16_t a0, fp16_t b0, fp32_t c) {
    return ampere_dp8a_fp32(&a0, &b0, 1, c);
}
static fp16_t dp16_1(fp16_t a0, fp16_t b0, fp16_t c) {
    return ampere_dp8a_fp16(&a0, &b0, 1, c);
}
static fp32_t dpbf_1(bf16_t a0, bf16_t b0, fp32_t c) {
    return ampere_dp8a_bf16(&a0, &b0, 1, c);
}

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

static void test_fp32_basic() {
    SECTION("FP32 accumulator – basic arithmetic (FP16 inputs)");

    CHECK_EQ_HEX(dp32_1(H_ONE, H_ONE, F_ZERO), F_ONE);

    CHECK_EQ_HEX(dp32_8(H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,
                        H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE, F_ZERO), F_EIGHT);

    CHECK_EQ_HEX(dp32_1(H_ONE, H_ONE, F_ONE), F_TWO);

    CHECK_EQ_HEX(dp32_1(H_NEG_ONE, H_ONE, F_ZERO), F_NEG_ONE);

    CHECK_EQ_HEX(dp32_1(H_ONE, H_NEG_ONE, F_TWO), F_ONE);

    {
        fp16_t a[1]={h(2.0f)}, bv[1]={h(3.0f)};
        CHECK_EQ_HEX(ampere_dp8a_fp32(a,bv,1,f32(4.0f)), f32(10.0f));
    }

    {
        fp16_t a[8]={h(1),h(2),h(3),h(4),h(5),h(6),h(7),h(8)};
        fp16_t bv[8]={H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE};
        CHECK_EQ_HEX(ampere_dp8a_fp32(a,bv,8,F_ZERO), f32(36.0f));
    }

    CHECK_EQ_HEX(dp32_1(H_NEG_ZERO, H_ONE, F_ZERO), F_ZERO);
}

static void test_fp32_specials() {
    SECTION("FP32 accumulator – special values");

    CHECK_EQ_HEX(dp32_1(H_NAN, H_ONE, F_ZERO), F_NAN_OUT);
    CHECK_EQ_HEX(dp32_1(H_ONE, H_NAN, F_ZERO), F_NAN_OUT);

    CHECK_EQ_HEX(dp32_1(H_ZERO, H_INF, F_ZERO), F_NAN_OUT);
    CHECK_EQ_HEX(dp32_1(H_INF,  H_ZERO, F_ZERO), F_NAN_OUT);

    CHECK_EQ_HEX(dp32_1(H_INF, H_ONE,     F_NEG_INF), F_NAN_OUT);
    CHECK_EQ_HEX(dp32_1(H_INF, H_NEG_ONE, F_INF),     F_NAN_OUT);

    CHECK_EQ_HEX(dp32_1(H_ONE, H_ONE, F_NAN_OUT), F_NAN_OUT);

    CHECK_EQ_HEX(dp32_1(H_INF, H_ONE, F_ZERO), F_INF);

    CHECK_EQ_HEX(dp32_1(H_INF, H_NEG_ONE, F_ZERO), F_NEG_INF);

    {
        fp32_t r = dp32_1(H_MAX, H_MAX, F_ZERO);
        CHECK(r != F_NAN_OUT && r != F_INF && r != F_NEG_INF);
    }
}

static void test_fp32_subnorm() {
    SECTION("FP32 accumulator – subnormals");

    {
        fp32_t r = dp32_1(H_SUBNORM1, H_ONE, F_ZERO);
        CHECK(r != F_ZERO);
    }

    CHECK_EQ_HEX(dp32_1(H_SUBNORM1, H_SUBNORM1, F_ZERO), (fp32_t)0x27800000u);
}

static void test_fp32_chaining() {
    SECTION("FP32 accumulator – multi-group chaining");

    {
        fp16_t a[16], bv[16];
        for (int i = 0; i < 16; ++i) { a[i] = H_ONE; bv[i] = H_ONE; }
        CHECK_EQ_HEX(ampere_dp8a_fp32(a,bv,16,F_ZERO), f32(16.0f));
    }

    {
        fp16_t a[9], bv[9];
        for (int i = 0; i < 9; ++i) { a[i] = H_ONE; bv[i] = H_ONE; }
        CHECK_EQ_HEX(ampere_dp8a_fp32(a,bv,9,F_ZERO), f32(9.0f));
    }

    {
        fp16_t a[7], bv[7];
        for (int i = 0; i < 7; ++i) { a[i] = h((float)(i+1)); bv[i] = h((float)(i+1)); }
        CHECK_EQ_HEX(ampere_dp8a_fp32(a,bv,7,F_ZERO), f32(140.0f));
    }
}

static void test_fp16_basic() {
    SECTION("FP16 accumulator – basic arithmetic");
    CHECK_EQ_HEX(dp16_1(H_ONE, H_ONE, H_ZERO), H_ONE);

    CHECK_EQ_HEX(dp16_8(H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,
                        H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE, H_ZERO),
                 h(8.0f));
    CHECK_EQ_HEX(dp16_1(H_ONE, H_ONE, H_ONE), H_TWO);

    CHECK_EQ_HEX(dp16_1(H_NEG_ONE, H_ONE, H_ZERO), H_NEG_ONE);

    CHECK_EQ_HEX(dp16_1(H_NEG_ZERO, H_ONE, H_ZERO), H_ZERO);
}

static void test_fp16_specials() {
    SECTION("FP16 accumulator – special values");

    CHECK_EQ_HEX(dp16_1(H_NAN, H_ONE, H_ZERO), H_NAN_OUT);
    CHECK_EQ_HEX(dp16_1(H_ONE, H_NAN, H_ZERO), H_NAN_OUT);

    CHECK_EQ_HEX(dp16_1(H_ZERO, H_INF, H_ZERO), H_NAN_OUT);
    CHECK_EQ_HEX(dp16_1(H_INF, H_ONE, H_NEG_INF), H_NAN_OUT);

    CHECK_EQ_HEX(dp16_1(H_ONE, H_ONE, H_NAN_OUT), H_NAN_OUT);
    CHECK_EQ_HEX(dp16_1(H_INF, H_ONE, H_ZERO), H_INF);

    CHECK_EQ_HEX(dp16_1(H_MAX, H_MAX, H_ZERO), H_INF);
}

static void test_fp16_rne() {
    SECTION("FP16 accumulator – RNE rounding");

    CHECK_EQ_HEX(dp16_1(H_HALF, H_HALF, H_ZERO), h(0.25f));

    CHECK_EQ_HEX(dp16_8(H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,
                        H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE,H_ONE, H_ZERO),
                 h(8.0f));
}

static void test_fp16_chaining() {
    SECTION("FP16 accumulator – multi-group chaining");

    {
        fp16_t a[16], bv[16];
        for (int i = 0; i < 16; ++i) { a[i] = H_ONE; bv[i] = H_ONE; }
        CHECK_EQ_HEX(ampere_dp8a_fp16(a,bv,16,H_ZERO), h(16.0f));
    }
}

static void test_bf16_basic() {
    SECTION("BF16 inputs, FP32 accumulator – basic arithmetic");

    CHECK_EQ_HEX(dpbf_1(BF_ONE, BF_ONE, F_ZERO), F_ONE);

    CHECK_EQ_HEX(dpbf_1(BF_ONE, BF_ONE, F_ONE), F_TWO);

    CHECK_EQ_HEX(dpbf_1(BF_NEG_ONE, BF_ONE, F_ZERO), F_NEG_ONE);

    {
        bf16_t a[1]={b(2.0f)}, bv[1]={b(3.0f)};
        CHECK_EQ_HEX(ampere_dp8a_bf16(a,bv,1,f32(4.0f)), f32(10.0f));
    }
    {
        bf16_t a[8], bv[8];
        for (int i = 0; i < 8; ++i) { a[i] = BF_ONE; bv[i] = BF_ONE; }
        CHECK_EQ_HEX(ampere_dp8a_bf16(a,bv,8,F_ZERO), f32(8.0f));
    }

    CHECK_EQ_HEX(dpbf_1(BF_NEG_ZERO, BF_ONE, F_ZERO), F_ZERO);
}

static void test_bf16_specials() {
    SECTION("BF16 inputs, FP32 accumulator – special values");

    CHECK_EQ_HEX(dpbf_1(BF_NAN, BF_ONE, F_ZERO), F_NAN_OUT);

    CHECK_EQ_HEX(dpbf_1(BF_ZERO, BF_INF, F_ZERO), F_NAN_OUT);

    CHECK_EQ_HEX(dpbf_1(BF_INF, BF_ONE, F_NEG_INF), F_NAN_OUT);

    CHECK_EQ_HEX(dpbf_1(BF_INF, BF_ONE, F_ZERO), F_INF);
}

static void test_bf16_chaining() {
    SECTION("BF16 inputs, FP32 accumulator – multi-group chaining");

    {
        bf16_t a[16], bv[16];
        for (int i = 0; i < 16; ++i) { a[i] = BF_ONE; bv[i] = BF_ONE; }
        CHECK_EQ_HEX(ampere_dp8a_bf16(a,bv,16,F_ZERO), f32(16.0f));
    }

    {
        bf16_t a[5], bv[5];
        for (int i = 0; i < 5; ++i) { a[i] = BF_TWO; bv[i] = BF_TWO; }
        CHECK_EQ_HEX(ampere_dp8a_bf16(a,bv,5,F_ZERO), f32(20.0f));
    }
}

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