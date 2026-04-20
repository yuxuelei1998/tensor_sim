#include "volta_tensor.h"
#include "fp_utils.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

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

static fp16_t h(float f)  { return float_to_fp16(f); }
static fp32_t f32(float f){ return float_to_bits(f); }
static float  tof(fp32_t b){ return bits_to_float(b); }

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
static constexpr fp16_t H_SUBNORM2 = 0x0002u;

static constexpr fp32_t F_ONE     = 0x3F800000u;
static constexpr fp32_t F_TWO     = 0x40000000u;
static constexpr fp32_t F_FOUR    = 0x40800000u;
static constexpr fp32_t F_ZERO    = 0x00000000u;
static constexpr fp32_t F_NEG_ONE = 0xBF800000u;
static constexpr fp32_t F_INF     = 0x7F800000u;
static constexpr fp32_t F_NEG_INF = 0xFF800000u;
static constexpr fp32_t F_NAN_OUT = 0x7FFFFFFFu;
static constexpr fp16_t H_NAN_OUT = 0x7FFFu;

#define SECTION(name)  std::printf("\n[%s]\n", name)

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

static void test_fp32_basic() {
    SECTION("FP32 accumulator – basic arithmetic");

    CHECK_EQ_HEX(dp32_1(H_ONE, H_ONE, F_ZERO), F_ONE);

    CHECK_EQ_HEX(dp32_4(H_ONE,H_ONE,H_ONE,H_ONE, H_ONE,H_ONE,H_ONE,H_ONE, F_ZERO), F_FOUR);

    CHECK_EQ_HEX(dp32_1(H_ONE, H_ONE, F_ONE), F_TWO);

    CHECK_EQ_HEX(dp32_1(H_NEG_ONE, H_ONE, F_ZERO), F_NEG_ONE);

    CHECK_EQ_HEX(dp32_1(H_ONE, H_NEG_ONE, F_TWO), F_ONE);

    {
        fp16_t a[1]={h(2.0f)}, b[1]={h(3.0f)};
        fp32_t res = volta_dp4a_fp32(a,b,1, f32(4.0f));
        CHECK_EQ_HEX(res, f32(10.0f));
    }

    {
        fp16_t a[4]={h(1),h(2),h(3),h(4)}, b[4]={H_ONE,H_ONE,H_ONE,H_ONE};
        CHECK_EQ_HEX(volta_dp4a_fp32(a,b,4,F_ZERO), f32(10.0f));
    }
    CHECK_EQ_HEX(dp32_1(H_NEG_ZERO, H_ONE, F_ZERO), F_ZERO);
}
static void test_fp32_specials() {
    SECTION("FP32 accumulator – special values");

    CHECK_EQ_HEX(dp32_1(H_NAN, H_ONE, F_ZERO), F_NAN_OUT);
    CHECK_EQ_HEX(dp32_1(H_ONE, H_NAN, F_ZERO), F_NAN_OUT);
    CHECK_EQ_HEX(dp32_1(H_ONE, H_ONE, 0x7FC00000u), F_NAN_OUT);

    CHECK_EQ_HEX(dp32_1(H_ZERO,    H_INF, F_ZERO), F_NAN_OUT);
    CHECK_EQ_HEX(dp32_1(H_INF,     H_ZERO, F_ZERO), F_NAN_OUT);
    CHECK_EQ_HEX(dp32_1(H_NEG_ZERO,H_INF, F_ZERO), F_NAN_OUT);
    CHECK_EQ_HEX(dp32_1(H_INF, H_ONE, F_NEG_INF), F_NAN_OUT);

    CHECK_EQ_HEX(dp32_1(H_INF, H_ONE, F_ZERO), F_INF);

    CHECK_EQ_HEX(dp32_1(H_INF, H_NEG_ONE, F_ZERO), F_NEG_INF);

    CHECK_EQ_HEX(dp32_1(H_ONE, H_ONE, F_INF), F_INF);
}

static void test_fp32_overflow() {
    SECTION("FP32 accumulator – overflow");
    {
        fp16_t a[1]={H_MAX}, b[1]={H_MAX};
        fp32_t res = volta_dp4a_fp32(a,b,1,F_ZERO);
        float v = tof(res);
        CHECK(v > 4.28e9f && v < 4.30e9f);
    }
    {
        fp16_t a[4]={H_MAX,H_MAX,H_MAX,H_MAX}, b[4]={H_MAX,H_MAX,H_MAX,H_MAX};
        fp32_t res = volta_dp4a_fp32(a,b,4,F_ZERO);
        float v = tof(res);
        CHECK(v > 1.7e10f && v < 1.73e10f);
    }

    {
        fp32_t fp32_max = 0x7F7FFFFFu;
        fp32_t res = dp32_1(H_MAX, H_MAX, fp32_max);
        CHECK_EQ_HEX(res, fp32_max);
    }

    {
        fp32_t big = 0x4F800000u;
        fp32_t res = dp32_1(H_ONE, H_ONE, big);
        CHECK(fp32_exp(res) != 0xFF);
    }
}

static void test_fp32_subnormal() {
    SECTION("FP32 accumulator – subnormal FP16 inputs");

    {
        fp32_t res = dp32_1(H_SUBNORM1, H_ONE, F_ZERO);
        float  v   = tof(res);
        float  exp = std::ldexpf(1.0f, -24);
        CHECK(std::fabsf(v - exp) < 1e-30f);
    }

    {
        fp32_t res = dp32_1(H_SUBNORM2, H_ONE, F_ZERO);
        float  v   = tof(res);
        float  exp = std::ldexpf(1.0f, -23);
        CHECK(std::fabsf(v - exp) < 1e-30f);
    }

    {
        fp32_t res = volta_dp4a_fp32(&H_SUBNORM1, &H_SUBNORM1, 1, F_ZERO);
        fp32_t expected = (79u << 23);
        CHECK_EQ_HEX(res, expected);
    }
    {
        fp32_t tiny_c = 0x00000001u;
        fp32_t res = dp32_1(H_ZERO, H_ZERO, tiny_c);
        CHECK_EQ_HEX(res, tiny_c);
    }
}

static void test_fp32_long_vectors() {
    SECTION("FP32 accumulator – long vectors");

    {
        fp16_t a[8], b[8];
        for(int i=0;i<8;++i){a[i]=H_ONE;b[i]=H_ONE;}
        fp32_t res = volta_dp4a_fp32(a,b,8,F_ZERO);
        CHECK_EQ_HEX(res, f32(8.0f));
    }

    {
        fp16_t a[5], b[5];
        for(int i=0;i<5;++i){a[i]=H_ONE;b[i]=H_ONE;}
        fp32_t res = volta_dp4a_fp32(a,b,5,F_ZERO);
        CHECK_EQ_HEX(res, f32(5.0f));
    }

    {
        fp16_t a[1]={H_ONE}, b[1]={H_ONE};
        CHECK_EQ_HEX(volta_dp4a_fp32(a,b,1,F_ZERO), F_ONE);
    }
    {
        fp16_t a[12], b[12];
        for(int i=0;i<12;++i){a[i]=H_ONE;b[i]=H_ONE;}
        CHECK_EQ_HEX(volta_dp4a_fp32(a,b,12,F_ZERO), f32(12.0f));
    }
}

static void test_fp16_basic() {
    SECTION("FP16 accumulator – basic arithmetic");

    CHECK_EQ_HEX(dp16_1(H_ONE, H_ONE, H_ZERO), H_ONE);

    CHECK_EQ_HEX(dp16_4(H_ONE,H_ONE,H_ONE,H_ONE, H_ONE,H_ONE,H_ONE,H_ONE, H_ZERO),
                 (fp16_t)0x4400u);

    CHECK_EQ_HEX(dp16_1(H_ONE, H_ONE, H_ONE), H_TWO);

    CHECK_EQ_HEX(dp16_1(H_NEG_ONE, H_ONE, H_ZERO), H_NEG_ONE);
    {
        fp16_t a[1]={h(2.0f)}, b[1]={h(3.0f)};
        fp16_t res = volta_dp4a_fp16(a,b,1, h(4.0f));
        CHECK_EQ_HEX(res, h(10.0f));
    }

    CHECK_EQ_HEX(dp16_1(H_NEG_ZERO, H_ONE, H_ZERO), H_ZERO);

    {
        fp16_t a[4]={h(1),h(2),h(3),h(4)}, b[4]={H_ONE,H_ONE,H_ONE,H_ONE};
        CHECK_EQ_HEX(volta_dp4a_fp16(a,b,4,H_ZERO), h(10.0f));
    }
}

static void test_fp16_specials() {
    SECTION("FP16 accumulator – special values");

    CHECK_EQ_HEX(dp16_1(H_NAN,  H_ONE,  H_ZERO), H_NAN_OUT);
    CHECK_EQ_HEX(dp16_1(H_ONE,  H_NAN,  H_ZERO), H_NAN_OUT);
    CHECK_EQ_HEX(dp16_1(H_ONE,  H_ONE,  H_NAN),  H_NAN_OUT);

    CHECK_EQ_HEX(dp16_1(H_ZERO, H_INF,  H_ZERO), H_NAN_OUT);
    CHECK_EQ_HEX(dp16_1(H_INF,  H_ZERO, H_ZERO), H_NAN_OUT);

    CHECK_EQ_HEX(dp16_1(H_INF,  H_ONE,  H_NEG_INF), H_NAN_OUT);

    CHECK_EQ_HEX(dp16_1(H_INF, H_ONE, H_ZERO), H_INF);

    CHECK_EQ_HEX(dp16_1(H_INF, H_NEG_ONE, H_ZERO), H_NEG_INF);

    CHECK_EQ_HEX(dp16_1(H_ONE, H_ONE, H_INF), H_INF);
}

static void test_fp16_overflow() {
    SECTION("FP16 accumulator – overflow");

    CHECK_EQ_HEX(dp16_1(H_MAX, H_TWO, H_ZERO), H_INF);
    CHECK_EQ_HEX(dp16_1(H_MAX, H_MAX, H_ZERO), H_INF);

    {
        fp16_t a[4]={H_MAX,H_MAX,H_MAX,H_MAX}, b[4]={H_ONE,H_ONE,H_ONE,H_ONE};
        fp16_t res = volta_dp4a_fp16(a,b,4,H_ZERO);
        CHECK_EQ_HEX(res, H_INF);
    }
}

static void test_fp16_subnormal() {
    SECTION("FP16 accumulator – subnormal inputs");
    {
        fp16_t res = dp16_1(H_SUBNORM1, H_ONE, H_ZERO);
        CHECK_EQ_HEX(res, H_SUBNORM1);
    }

    {
        fp16_t res = dp16_1(H_SUBNORM2, H_ONE, H_ZERO);
        CHECK_EQ_HEX(res, H_SUBNORM2);
    }
    {
        fp16_t res = dp16_1(H_SUBNORM1, H_SUBNORM1, H_ZERO);
        CHECK_EQ_HEX(res, H_ZERO);
    }
    {
        fp16_t sub_c = 0x0003u;
        fp16_t res = dp16_1(H_ZERO, H_ZERO, sub_c);
        CHECK_EQ_HEX(res, sub_c);
    }
}

static void test_fp16_long_vectors() {
    SECTION("FP16 accumulator – long vectors");

    {
        fp16_t a[8], b[8];
        for(int i=0;i<8;++i){a[i]=H_ONE;b[i]=H_ONE;}
        CHECK_EQ_HEX(volta_dp4a_fp16(a,b,8,H_ZERO), h(8.0f));
    }

    {
        fp16_t a[5], b[5];
        for(int i=0;i<5;++i){a[i]=H_ONE;b[i]=H_ONE;}
        CHECK_EQ_HEX(volta_dp4a_fp16(a,b,5,H_ZERO), h(5.0f));
    }
}

static void test_fp16_rne() {
    SECTION("FP16 accumulator – RNE rounding");

    {
        fp16_t a_val = 0x3800u;
        fp16_t b_val = 0x1400u;
        fp16_t res = dp16_1(a_val, b_val, H_ONE);
        CHECK_EQ_HEX(res, H_ONE);
        std::printf("    RNE tie (even mantissa=0): 1.0 + 2^-11 → 0x%04X (expect 0x3C00)\n", res);
    }

    {
        fp16_t c_odd = 0x3C01u;
        fp16_t a_val = 0x3800u;
        fp16_t b_val = 0x1400u;
        fp16_t res = dp16_1(a_val, b_val, c_odd);
        CHECK_EQ_HEX(res, (fp16_t)0x3C02u);
        std::printf("    RNE tie (odd mantissa=1): → 0x%04X (expect 0x3C02)\n", res);
    }

    {
        fp16_t a[4]={0x3800u, H_SUBNORM1, H_ZERO, H_ZERO};
        fp16_t b[4]={0x1400u, H_ONE,      H_ZERO, H_ZERO};
        fp16_t res = volta_dp4a_fp16(a,b,4,H_ONE);
        CHECK_EQ_HEX(res, (fp16_t)0x3C00u);
        std::printf("    RNE sticky=0 (alignment bits discarded): 1.0 + 2^-11 → 0x%04X (expect 0x3C00)\n", res);
    }
}

static void test_fp32_cancel() {
    SECTION("FP32 accumulator – cancellation");

    CHECK_EQ_HEX(dp32_1(H_ONE, H_ONE, F_NEG_ONE), F_ZERO);

    {
        fp16_t a[1]={H_TWO}, b[1]={H_TWO};
        CHECK_EQ_HEX(volta_dp4a_fp32(a,b,1, f32(-4.0f)), F_ZERO);
    }

    CHECK_EQ_HEX(dp32_1(H_NEG_ONE, H_NEG_ONE, F_ZERO), F_ONE);

    {
        fp16_t a[4]={H_ONE,H_ONE,H_ONE,H_ONE};
        fp16_t b[4]={H_NEG_ONE,H_NEG_ONE,H_NEG_ONE,H_NEG_ONE};
        CHECK_EQ_HEX(volta_dp4a_fp32(a,b,4,F_ZERO), f32(-4.0f));
    }
}

static void test_fp16_cancel() {
    SECTION("FP16 accumulator – cancellation");

    CHECK_EQ_HEX(dp16_1(H_ONE, H_ONE, H_NEG_ONE), H_ZERO);

    CHECK_EQ_HEX(dp16_1(H_NEG_ONE, H_NEG_ONE, H_ZERO), H_ONE);

    {
        fp16_t a[2]={H_TWO,H_TWO}, b[2]={H_NEG_ONE,H_NEG_ONE};
        CHECK_EQ_HEX(volta_dp4a_fp16(a,b,2,H_ZERO), h(-4.0f));
    }
}

static void test_fp32_cross_check() {
    SECTION("FP32 accumulator – cross-check vs float");

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