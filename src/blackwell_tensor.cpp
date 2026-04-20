#include "blackwell_tensor.h"
#include <algorithm>
#include <climits>

static constexpr fp32_t NAN_OUT_FP32 = 0x7FFFFFFFu;
static constexpr fp16_t NAN_OUT_FP16 = 0x7FFFu;

struct IntermVal {
    enum Kind : uint8_t { ZERO, NORMAL, POS_INF, NEG_INF, POS_INF_OVF, NEG_INF_OVF, NAN_VAL } kind = ZERO;
    bool     sign = false;
    int32_t  exp  = 0;
    uint32_t mant = 0;
};

struct FP16Norm { bool sign; int32_t exp; uint32_t sig; };
struct BF16Norm { bool sign; int32_t exp; uint32_t sig; };
struct E5M2Norm { bool sign; int32_t exp; uint32_t sig; };
struct E4M3Norm { bool sign; int32_t exp; uint32_t sig; };

static FP16Norm decode_fp16(fp16_t x) {
    FP16Norm r{};
    r.sign = fp16_sign(x);
    int      e = fp16_exp(x);
    uint32_t m = fp16_mant(x);
    if (e == 0) {
        int lz = __builtin_clz(m) - 22;
        r.exp  = -15 - lz;
        r.sig  = m << (lz + 1);
    } else {
        r.exp = e - 15;
        r.sig = (1u << 10) | m;
    }
    return r;
}

static BF16Norm decode_bf16(bf16_t x) {
    BF16Norm r{};
    r.sign = bf16_sign(x);
    int      e = bf16_exp(x);
    uint32_t m = bf16_mant(x);
    if (e == 0) {
        int lz = __builtin_clz(m) - 25;
        r.exp  = -127 - lz;
        r.sig  = m << (lz + 1);
    } else {
        r.exp = e - 127;
        r.sig = (1u << 7) | m;
    }
    return r;
}

static E5M2Norm decode_e5m2(e5m2_t x) {
    E5M2Norm r{};
    r.sign = e5m2_sign(x);
    int      e = e5m2_exp(x);
    uint32_t m = e5m2_mant(x);
    if (e == 0) {
        int lead = (m >= 2) ? 1 : 0;
        r.sig = m << (2 - lead);
        r.exp = lead - 16;
    } else {
        r.exp = e - 15;
        r.sig = (1u << 2) | m;
    }
    return r;
}

static E4M3Norm decode_e4m3(e4m3_t x) {
    E4M3Norm r{};
    r.sign = e4m3_sign(x);
    int      e = e4m3_exp(x);
    uint32_t m = e4m3_mant(x);
    if (e == 0) {
        int lead_pos = (m != 0) ? (31 - __builtin_clz(m)) : -1;
        r.sig = m << (3 - lead_pos);
        r.exp = lead_pos - 9;
    } else {
        r.exp = e - 7;
        r.sig = (1u << 3) | m;
    }
    return r;
}

struct E2M1Norm { bool sign; int32_t exp; uint32_t sig; };

static E2M1Norm decode_e2m1(e2m1_t x) {
    E2M1Norm r{};
    r.sign = e2m1_sign(x);
    int      e = e2m1_exp(x);
    uint32_t m = e2m1_mant(x);
    if (e == 0) {
        r.sig = m << 1;
        r.exp = -1;
    } else {
        r.exp = e - 1;
        r.sig = (1u << 1) | m;
    }
    return r;
}

static void mul11_25(uint32_t sig_a, int32_t exp_a,
                     uint32_t sig_b, int32_t exp_b,
                     int32_t& out_exp, uint32_t& out_mant) {
    uint32_t prod = sig_a * sig_b;
    if (prod >= (1u << 21)) {
        out_mant = (prod - (1u << 21)) << 4;
        out_exp  = exp_a + exp_b + 1;
    } else {
        out_mant = (prod - (1u << 20)) << 5;
        out_exp  = exp_a + exp_b;
    }
}

static void mul8_25(uint32_t sig_a, int32_t exp_a,
                    uint32_t sig_b, int32_t exp_b,
                    int32_t& out_exp, uint32_t& out_mant) {
    uint32_t prod = sig_a * sig_b;
    if (prod >= (1u << 15)) {
        out_mant = (prod - (1u << 15)) << 10;
        out_exp  = exp_a + exp_b + 1;
    } else {
        out_mant = (prod - (1u << 14)) << 11;
        out_exp  = exp_a + exp_b;
    }
}

static void mul3_25(uint32_t sig_a, int32_t exp_a,
                    uint32_t sig_b, int32_t exp_b,
                    int32_t& out_exp, uint32_t& out_mant) {
    uint32_t prod = sig_a * sig_b;
    if (prod >= (1u << 5)) {

        out_mant = (prod - (1u << 5)) << 20;
        out_exp  = exp_a + exp_b + 1;
    } else {

        out_mant = (prod - (1u << 4)) << 21;
        out_exp  = exp_a + exp_b;
    }
}

static void mul4_25(uint32_t sig_a, int32_t exp_a,
                    uint32_t sig_b, int32_t exp_b,
                    int32_t& out_exp, uint32_t& out_mant) {
    uint32_t prod = sig_a * sig_b;
    if (prod >= (1u << 7)) {

        out_mant = (prod - (1u << 7)) << 18;
        out_exp  = exp_a + exp_b + 1;
    } else {

        out_mant = (prod - (1u << 6)) << 19;
        out_exp  = exp_a + exp_b;
    }
}

static void mul2_25(uint32_t sig_a, int32_t exp_a,
                    uint32_t sig_b, int32_t exp_b,
                    int32_t& out_exp, uint32_t& out_mant) {
    uint32_t prod = sig_a * sig_b;
    if (prod >= (1u << 3)) {
        out_mant = (prod - (1u << 3)) << 22;
        out_exp  = exp_a + exp_b + 1;
    } else {
        out_mant = (prod - (1u << 2)) << 23;
        out_exp  = exp_a + exp_b;
    }
}

static IntermVal fp16_mul_fp32acc_25(fp16_t a, fp16_t b) {
    bool rs = fp16_sign(a) ^ fp16_sign(b);
    if (fp16_is_nan(a) || fp16_is_nan(b))                    return {IntermVal::NAN_VAL};
    if ((fp16_is_zero(a) && fp16_is_inf(b)) ||
        (fp16_is_inf(a)  && fp16_is_zero(b)))                return {IntermVal::NAN_VAL};
    if (fp16_is_zero(a) || fp16_is_zero(b))                  return {IntermVal::ZERO};
    if (fp16_is_inf(a)  || fp16_is_inf(b))
        return {rs ? IntermVal::NEG_INF : IntermVal::POS_INF, rs};
    FP16Norm na = decode_fp16(a), nb = decode_fp16(b);
    int32_t e; uint32_t m;
    mul11_25(na.sig, na.exp, nb.sig, nb.exp, e, m);
    return {IntermVal::NORMAL, rs, e, m};
}

static IntermVal fp16_mul_fp16acc_25(fp16_t a, fp16_t b) {
    bool rs = fp16_sign(a) ^ fp16_sign(b);
    if (fp16_is_nan(a) || fp16_is_nan(b))                    return {IntermVal::NAN_VAL};
    if ((fp16_is_zero(a) && fp16_is_inf(b)) ||
        (fp16_is_inf(a)  && fp16_is_zero(b)))                return {IntermVal::NAN_VAL};
    if (fp16_is_zero(a) || fp16_is_zero(b))                  return {IntermVal::ZERO};
    if (fp16_is_inf(a)  || fp16_is_inf(b))
        return {rs ? IntermVal::NEG_INF : IntermVal::POS_INF, rs};
    FP16Norm na = decode_fp16(a), nb = decode_fp16(b);
    int32_t e; uint32_t m;
    mul11_25(na.sig, na.exp, nb.sig, nb.exp, e, m);
    if (e > 15)  return {rs ? IntermVal::NEG_INF_OVF : IntermVal::POS_INF_OVF, rs};
    if (e >= -14) return {IntermVal::NORMAL, rs, e, m};
    int shift = -14 - e;
    static constexpr int SHIFT_CAP = 26;
    if (shift >= SHIFT_CAP) return {IntermVal::ZERO};
    uint64_t full_sig = ((uint64_t)1 << 25) | m;
    uint64_t sub_sig  = full_sig >> shift;
    if (sub_sig == 0) return {IntermVal::ZERO};
    int lead = 63 - __builtin_clzll(sub_sig);
    int32_t new_exp = lead - 39;
    uint32_t new_mant;
    if (lead >= 25) new_mant = (uint32_t)((sub_sig >> (lead - 25)) & 0x1FFFFFFu);
    else            new_mant = (uint32_t)((sub_sig << (25 - lead)) & 0x1FFFFFFu);
    return {IntermVal::NORMAL, rs, new_exp, new_mant};
}

static IntermVal bf16_mul_fp32acc_25(bf16_t a, bf16_t b) {
    bool rs = bf16_sign(a) ^ bf16_sign(b);
    if (bf16_is_nan(a) || bf16_is_nan(b))                    return {IntermVal::NAN_VAL};
    if ((bf16_is_zero(a) && bf16_is_inf(b)) ||
        (bf16_is_inf(a)  && bf16_is_zero(b)))                return {IntermVal::NAN_VAL};
    if (bf16_is_zero(a) || bf16_is_zero(b))                  return {IntermVal::ZERO};
    if (bf16_is_inf(a)  || bf16_is_inf(b))
        return {rs ? IntermVal::NEG_INF : IntermVal::POS_INF, rs};
    BF16Norm na = decode_bf16(a), nb = decode_bf16(b);
    int32_t e; uint32_t m;
    mul8_25(na.sig, na.exp, nb.sig, nb.exp, e, m);
    return {IntermVal::NORMAL, rs, e, m};
}

static IntermVal e5m2_mul_fp32acc_25(e5m2_t a, e5m2_t b) {
    bool rs = e5m2_sign(a) ^ e5m2_sign(b);
    if (e5m2_is_nan(a) || e5m2_is_nan(b))                    return {IntermVal::NAN_VAL};
    if ((e5m2_is_zero(a) && e5m2_is_inf(b)) ||
        (e5m2_is_inf(a)  && e5m2_is_zero(b)))                return {IntermVal::NAN_VAL};
    if (e5m2_is_zero(a) || e5m2_is_zero(b))                  return {IntermVal::ZERO};
    if (e5m2_is_inf(a)  || e5m2_is_inf(b))
        return {rs ? IntermVal::NEG_INF : IntermVal::POS_INF, rs};
    E5M2Norm na = decode_e5m2(a), nb = decode_e5m2(b);
    int32_t e; uint32_t m;
    mul3_25(na.sig, na.exp, nb.sig, nb.exp, e, m);
    return {IntermVal::NORMAL, rs, e, m};
}

static IntermVal e5m2_mul_fp16acc_25(e5m2_t a, e5m2_t b) {
    bool rs = e5m2_sign(a) ^ e5m2_sign(b);
    if (e5m2_is_nan(a) || e5m2_is_nan(b))                    return {IntermVal::NAN_VAL};
    if ((e5m2_is_zero(a) && e5m2_is_inf(b)) ||
        (e5m2_is_inf(a)  && e5m2_is_zero(b)))                return {IntermVal::NAN_VAL};
    if (e5m2_is_zero(a) || e5m2_is_zero(b))                  return {IntermVal::ZERO};
    if (e5m2_is_inf(a)  || e5m2_is_inf(b))
        return {rs ? IntermVal::NEG_INF : IntermVal::POS_INF, rs};
    E5M2Norm na = decode_e5m2(a), nb = decode_e5m2(b);
    int32_t e; uint32_t m;
    mul3_25(na.sig, na.exp, nb.sig, nb.exp, e, m);
    if (e > 15)  return {rs ? IntermVal::NEG_INF_OVF : IntermVal::POS_INF_OVF, rs};
    if (e >= -14) return {IntermVal::NORMAL, rs, e, m};

    int shift = -14 - e;
    static constexpr int SHIFT_CAP = 26;
    if (shift >= SHIFT_CAP) return {IntermVal::ZERO};
    uint64_t full_sig = ((uint64_t)1 << 25) | m;
    uint64_t sub_sig  = full_sig >> shift;
    if (sub_sig == 0) return {IntermVal::ZERO};
    int lead = 63 - __builtin_clzll(sub_sig);
    int32_t new_exp = lead - 39;
    uint32_t new_mant;
    if (lead >= 25) new_mant = (uint32_t)((sub_sig >> (lead - 25)) & 0x1FFFFFFu);
    else            new_mant = (uint32_t)((sub_sig << (25 - lead)) & 0x1FFFFFFu);
    return {IntermVal::NORMAL, rs, new_exp, new_mant};
}

static IntermVal e4m3_mul_fp32acc_25(e4m3_t a, e4m3_t b) {
    bool rs = e4m3_sign(a) ^ e4m3_sign(b);
    if (e4m3_is_nan(a) || e4m3_is_nan(b))                    return {IntermVal::NAN_VAL};
    if (e4m3_is_zero(a) || e4m3_is_zero(b))                  return {IntermVal::ZERO};
    E4M3Norm na = decode_e4m3(a), nb = decode_e4m3(b);
    int32_t e; uint32_t m;
    mul4_25(na.sig, na.exp, nb.sig, nb.exp, e, m);
    return {IntermVal::NORMAL, rs, e, m};
}

static IntermVal e4m3_mul_fp16acc_25(e4m3_t a, e4m3_t b) {
    bool rs = e4m3_sign(a) ^ e4m3_sign(b);
    if (e4m3_is_nan(a) || e4m3_is_nan(b))                    return {IntermVal::NAN_VAL};
    if (e4m3_is_zero(a) || e4m3_is_zero(b))                  return {IntermVal::ZERO};
    E4M3Norm na = decode_e4m3(a), nb = decode_e4m3(b);
    int32_t e; uint32_t m;
    mul4_25(na.sig, na.exp, nb.sig, nb.exp, e, m);
    if (e > 15)  return {rs ? IntermVal::NEG_INF_OVF : IntermVal::POS_INF_OVF, rs};
    if (e >= -14) return {IntermVal::NORMAL, rs, e, m};

    int shift = -14 - e;
    static constexpr int SHIFT_CAP = 26;
    if (shift >= SHIFT_CAP) return {IntermVal::ZERO};
    uint64_t full_sig = ((uint64_t)1 << 25) | m;
    uint64_t sub_sig  = full_sig >> shift;
    if (sub_sig == 0) return {IntermVal::ZERO};
    int lead = 63 - __builtin_clzll(sub_sig);
    int32_t new_exp = lead - 39;
    uint32_t new_mant;
    if (lead >= 25) new_mant = (uint32_t)((sub_sig >> (lead - 25)) & 0x1FFFFFFu);
    else            new_mant = (uint32_t)((sub_sig << (25 - lead)) & 0x1FFFFFFu);
    return {IntermVal::NORMAL, rs, new_exp, new_mant};
}

static IntermVal e2m1_mul_fp32acc_25(e2m1_t a, e2m1_t b) {
    bool rs = e2m1_sign(a) ^ e2m1_sign(b);
    if (e2m1_is_zero(a) || e2m1_is_zero(b))                  return {IntermVal::ZERO};
    E2M1Norm na = decode_e2m1(a), nb = decode_e2m1(b);
    int32_t e; uint32_t m;
    mul2_25(na.sig, na.exp, nb.sig, nb.exp, e, m);
    return {IntermVal::NORMAL, rs, e, m};
}

static IntermVal e2m1_mul_fp16acc_25(e2m1_t a, e2m1_t b) {
    bool rs = e2m1_sign(a) ^ e2m1_sign(b);
    if (e2m1_is_zero(a) || e2m1_is_zero(b))                  return {IntermVal::ZERO};
    E2M1Norm na = decode_e2m1(a), nb = decode_e2m1(b);
    int32_t e; uint32_t m;
    mul2_25(na.sig, na.exp, nb.sig, nb.exp, e, m);
    if (e > 15)  return {rs ? IntermVal::NEG_INF_OVF : IntermVal::POS_INF_OVF, rs};
    if (e >= -14) return {IntermVal::NORMAL, rs, e, m};

    int shift = -14 - e;
    static constexpr int SHIFT_CAP = 26;
    if (shift >= SHIFT_CAP) return {IntermVal::ZERO};
    uint64_t full_sig = ((uint64_t)1 << 25) | m;
    uint64_t sub_sig  = full_sig >> shift;
    if (sub_sig == 0) return {IntermVal::ZERO};
    int lead = 63 - __builtin_clzll(sub_sig);
    int32_t new_exp = lead - 39;
    uint32_t new_mant;
    if (lead >= 25) new_mant = (uint32_t)((sub_sig >> (lead - 25)) & 0x1FFFFFFu);
    else            new_mant = (uint32_t)((sub_sig << (25 - lead)) & 0x1FFFFFFu);
    return {IntermVal::NORMAL, rs, new_exp, new_mant};
}

static IntermVal fp32_to_intermed25(fp32_t c) {
    bool sc = fp32_sign(c);
    int  e  = fp32_exp(c);
    uint32_t m = fp32_mant(c);
    if (e == 0xFF && m != 0) return {IntermVal::NAN_VAL};
    if (e == 0xFF && m == 0) return {sc ? IntermVal::NEG_INF : IntermVal::POS_INF, sc};
    if (fp32_is_zero(c))     return {IntermVal::ZERO};
    if (e == 0) {

        int lz = __builtin_clz(m) - 9;
        int p  = 22 - lz;
        int32_t  unb_exp = p - 149;
        uint32_t mant25  = (m - (1u << p)) << (25 - p);
        return {IntermVal::NORMAL, sc, unb_exp, mant25};
    }

    return {IntermVal::NORMAL, sc, e - 127, m << 2};
}

static IntermVal fp16_to_intermed25(fp16_t c) {
    bool sc = fp16_sign(c);
    int  e  = fp16_exp(c);
    uint32_t m = fp16_mant(c);
    if (e == 31 && m != 0) return {IntermVal::NAN_VAL};
    if (e == 31 && m == 0) return {sc ? IntermVal::NEG_INF : IntermVal::POS_INF, sc};
    if (fp16_is_zero(c))   return {IntermVal::ZERO};
    if (e == 0) {

        int lz  = __builtin_clz(m) - 22;
        int unb = -15 - lz;
        uint32_t sig11  = m << (lz + 1);
        uint32_t mant25 = (sig11 & 0x3FFu) << 15;
        return {IntermVal::NORMAL, sc, unb, mant25};
    }

    return {IntermVal::NORMAL, sc, e - 15, m << 15};
}

template<typename T>
static bool check_specials_n(const IntermVal* v, int n,
                              T nan_out, T pos_inf_out, T neg_inf_out, T& out) {

    for (int i = 0; i < n; ++i)
        if (v[i].kind == IntermVal::NAN_VAL) { out = nan_out; return true; }

    bool true_pos = false, true_neg = false;
    for (int i = 0; i < n; ++i) {
        if (v[i].kind == IntermVal::POS_INF) true_pos = true;
        if (v[i].kind == IntermVal::NEG_INF) true_neg = true;
    }

    if (true_pos && true_neg) { out = nan_out;     return true; }

    if (true_pos)             { out = pos_inf_out; return true; }
    if (true_neg)             { out = neg_inf_out; return true; }

    bool ovf_pos = false, ovf_neg = false;
    for (int i = 0; i < n; ++i) {
        if (v[i].kind == IntermVal::POS_INF_OVF) ovf_pos = true;
        if (v[i].kind == IntermVal::NEG_INF_OVF) ovf_neg = true;
    }
    if (ovf_pos && ovf_neg) { out = nan_out;     return true; }
    if (ovf_pos)            { out = pos_inf_out; return true; }
    if (ovf_neg)            { out = neg_inf_out; return true; }

    return false;
}

static void align_accum_n(const IntermVal* v, int n,
                           bool& sign_out, uint64_t& abs_out, int32_t& max_exp_out) {
    static constexpr int MANT_BITS  = 25;
    static constexpr int SHIFT_CAP  = MANT_BITS + 1;

    int32_t max_exp = INT32_MIN;
    for (int i = 0; i < n; ++i)
        if (v[i].kind == IntermVal::NORMAL)
            max_exp = std::max(max_exp, v[i].exp);
    max_exp_out = max_exp;

    if (max_exp == INT32_MIN) { sign_out = false; abs_out = 0; return; }

    int64_t accum = 0;
    for (int i = 0; i < n; ++i) {
        if (v[i].kind != IntermVal::NORMAL) continue;
        int shift = max_exp - v[i].exp;
        if (shift >= SHIFT_CAP) continue;
        uint64_t sig     = ((uint64_t)1 << MANT_BITS) | v[i].mant;
        uint64_t aligned = sig >> shift;
        if (v[i].sign) accum -= (int64_t)aligned;
        else           accum += (int64_t)aligned;
    }

    sign_out = accum < 0;
    abs_out  = sign_out ? (uint64_t)(-accum) : (uint64_t)accum;
}

static fp32_t accumulate_fp32_trunc(const IntermVal* v, int n) {
    fp32_t sig_out;
    if (check_specials_n<fp32_t>(v, n, NAN_OUT_FP32, 0x7F800000u, 0xFF800000u, sig_out))
        return sig_out;

    bool rs; uint64_t abs_sum; int32_t max_exp;
    align_accum_n(v, n, rs, abs_sum, max_exp);
    if (abs_sum == 0) return 0u;

    static constexpr int MANT_BITS = 25;
    int lead    = 63 - __builtin_clzll(abs_sum);
    int32_t unb = lead + max_exp - MANT_BITS;
    int32_t bsd = unb + 127;

    uint32_t mant23;
    if (lead >= 23) mant23 = (uint32_t)((abs_sum >> (lead - 23)) & 0x7FFFFFu);
    else            mant23 = (uint32_t)((abs_sum << (23 - lead)) & 0x7FFFFFu);

    if (bsd >= 255) return rs ? 0xFF800000u : 0x7F800000u;

    if (bsd <= 0) {
        int sub_shift = 1 - bsd;
        if (sub_shift >= 24) return (uint32_t)rs << 31;
        uint32_t sub_mant = ((1u << 23) | mant23) >> sub_shift;
        return ((uint32_t)rs << 31) | sub_mant;
    }

    return ((uint32_t)rs << 31) | ((uint32_t)bsd << 23) | mant23;
}

static fp16_t accumulate_fp16_rne(const IntermVal* v, int n) {
    fp16_t sig_out;
    if (check_specials_n<fp16_t>(v, n, NAN_OUT_FP16, 0x7C00u, 0xFC00u, sig_out))
        return sig_out;

    bool rs; uint64_t abs_sum; int32_t max_exp;
    align_accum_n(v, n, rs, abs_sum, max_exp);
    if (abs_sum == 0) return 0u;

    static constexpr int MANT_BITS = 25;
    int lead    = 63 - __builtin_clzll(abs_sum);
    int32_t unb = lead + max_exp - MANT_BITS;

    uint32_t mantX; bool sticky = false;
    if (lead >= MANT_BITS) {
        int rs2 = lead - MANT_BITS;
        mantX   = (uint32_t)((abs_sum >> rs2) & ((1u << MANT_BITS) - 1));
        sticky  = (rs2 > 0) && ((abs_sum & ((1ULL << rs2) - 1)) != 0);
    } else {
        mantX = (uint32_t)((abs_sum << (MANT_BITS - lead)) & ((1u << MANT_BITS) - 1));
    }

    const int RND_SHIFT = MANT_BITS - 10;

    auto rne_round_up = [](uint32_t mant10, int round_b, bool stk) -> bool {
        if (!round_b) return false;
        if (stk)      return true;
        return (mant10 & 1u) != 0;
    };

    if (unb >= -14) {
        if (unb > 15) goto overflow_inf;
        uint32_t mant10  = mantX >> RND_SHIFT;
        int      rnd_bit = (mantX >> (RND_SHIFT - 1)) & 1;
        bool     stk     = ((mantX & ((1u << (RND_SHIFT - 1)) - 1)) != 0) || sticky;
        if (rne_round_up(mant10, rnd_bit, stk)) {
            mant10++;
            if (mant10 >= (1u << 10)) { mant10 = 0; unb++; }
        }
        if (unb > 15) goto overflow_inf;
        uint32_t e16 = (uint32_t)(unb + 15);
        return (fp16_t)(((uint32_t)rs << 15) | (e16 << 10) | mant10);
    }

    {
        int shift_s = MANT_BITS - 24 - unb;
        uint64_t sig_full = ((uint64_t)1 << MANT_BITS) | mantX;

        if (shift_s > MANT_BITS) {
            int rnd_pos = shift_s - 1;
            int rnd_bit = (rnd_pos <= MANT_BITS) ? (int)((sig_full >> rnd_pos) & 1) : 0;
            bool stk2   = sticky || (rnd_pos > 0 && rnd_pos <= MANT_BITS &&
                                     ((sig_full & ((1ULL << rnd_pos) - 1)) != 0));
            bool ru = rne_round_up(0u, rnd_bit, stk2);
            if (!ru) return (fp16_t)((uint32_t)rs << 15);
            return (fp16_t)(((uint32_t)rs << 15) | 1u);
        }

        uint32_t mant10_s = (uint32_t)(sig_full >> shift_s) & 0x3FFu;
        int rnd_bit_s = (shift_s > 0) ? (int)((sig_full >> (shift_s - 1)) & 1) : 0;
        bool stk_s = sticky || (shift_s > 1 &&
                                ((sig_full & ((1ULL << (shift_s - 1)) - 1)) != 0));
        if (rne_round_up(mant10_s, rnd_bit_s, stk_s)) {
            mant10_s++;
            if (mant10_s >= (1u << 10))
                return (fp16_t)(((uint32_t)rs << 15) | (1u << 10));
        }
        if (mant10_s == 0) return (fp16_t)((uint32_t)rs << 15);
        return (fp16_t)(((uint32_t)rs << 15) | mant10_s);
    }

overflow_inf:
    return (fp16_t)(((uint32_t)rs << 15) | 0x7C00u);
}

static fp32_t bw_dp16a_fp32_group(const fp16_t a[16], const fp16_t b[16], fp32_t c) {
    for (int i = 0; i < 16; ++i) {
        if (fp16_is_nan(a[i]) || fp16_is_nan(b[i]))   return NAN_OUT_FP32;
        if (fp16_is_zero(a[i]) && fp16_is_inf(b[i]))  return NAN_OUT_FP32;
        if (fp16_is_inf(a[i])  && fp16_is_zero(b[i])) return NAN_OUT_FP32;
    }
    if (fp32_is_nan(c)) return NAN_OUT_FP32;

    IntermVal v[17];
    for (int i = 0; i < 16; ++i) v[i] = fp16_mul_fp32acc_25(a[i], b[i]);
    v[16] = fp32_to_intermed25(c);
    return accumulate_fp32_trunc(v, 17);
}

static fp16_t bw_dp16a_fp16_group(const fp16_t a[16], const fp16_t b[16], fp16_t c) {
    for (int i = 0; i < 16; ++i) {
        if (fp16_is_nan(a[i]) || fp16_is_nan(b[i]))   return NAN_OUT_FP16;
        if (fp16_is_zero(a[i]) && fp16_is_inf(b[i]))  return NAN_OUT_FP16;
        if (fp16_is_inf(a[i])  && fp16_is_zero(b[i])) return NAN_OUT_FP16;
    }
    if (fp16_is_nan(c)) return NAN_OUT_FP16;

    IntermVal v[17];
    for (int i = 0; i < 16; ++i) v[i] = fp16_mul_fp16acc_25(a[i], b[i]);
    v[16] = fp16_to_intermed25(c);
    return accumulate_fp16_rne(v, 17);
}

static fp32_t bw_dp16a_bf16_group(const bf16_t a[16], const bf16_t b[16], fp32_t c) {
    for (int i = 0; i < 16; ++i) {
        if (bf16_is_nan(a[i]) || bf16_is_nan(b[i]))   return NAN_OUT_FP32;
        if (bf16_is_zero(a[i]) && bf16_is_inf(b[i]))  return NAN_OUT_FP32;
        if (bf16_is_inf(a[i])  && bf16_is_zero(b[i])) return NAN_OUT_FP32;
    }
    if (fp32_is_nan(c)) return NAN_OUT_FP32;

    IntermVal v[17];
    for (int i = 0; i < 16; ++i) v[i] = bf16_mul_fp32acc_25(a[i], b[i]);
    v[16] = fp32_to_intermed25(c);
    return accumulate_fp32_trunc(v, 17);
}

static fp32_t bw_dp32a_e5m2_fp32_group(const e5m2_t a[32], const e5m2_t b[32], fp32_t c) {
    for (int i = 0; i < 32; ++i) {
        if (e5m2_is_nan(a[i]) || e5m2_is_nan(b[i]))        return NAN_OUT_FP32;
        if (e5m2_is_zero(a[i]) && e5m2_is_inf(b[i]))       return NAN_OUT_FP32;
        if (e5m2_is_inf(a[i])  && e5m2_is_zero(b[i]))      return NAN_OUT_FP32;
    }
    if (fp32_is_nan(c)) return NAN_OUT_FP32;

    IntermVal v[33];
    for (int i = 0; i < 32; ++i) v[i] = e5m2_mul_fp32acc_25(a[i], b[i]);
    v[32] = fp32_to_intermed25(c);
    return accumulate_fp32_trunc(v, 33);
}

static fp16_t bw_dp32a_e5m2_fp16_group(const e5m2_t a[32], const e5m2_t b[32], fp16_t c) {
    for (int i = 0; i < 32; ++i) {
        if (e5m2_is_nan(a[i]) || e5m2_is_nan(b[i]))        return NAN_OUT_FP16;
        if (e5m2_is_zero(a[i]) && e5m2_is_inf(b[i]))       return NAN_OUT_FP16;
        if (e5m2_is_inf(a[i])  && e5m2_is_zero(b[i]))      return NAN_OUT_FP16;
    }
    if (fp16_is_nan(c)) return NAN_OUT_FP16;

    IntermVal v[33];
    for (int i = 0; i < 32; ++i) v[i] = e5m2_mul_fp16acc_25(a[i], b[i]);
    v[32] = fp16_to_intermed25(c);
    return accumulate_fp16_rne(v, 33);
}

static fp32_t bw_dp32a_e4m3_fp32_group(const e4m3_t a[32], const e4m3_t b[32], fp32_t c) {
    for (int i = 0; i < 32; ++i) {
        if (e4m3_is_nan(a[i]) || e4m3_is_nan(b[i]))        return NAN_OUT_FP32;
    }
    if (fp32_is_nan(c)) return NAN_OUT_FP32;

    IntermVal v[33];
    for (int i = 0; i < 32; ++i) v[i] = e4m3_mul_fp32acc_25(a[i], b[i]);
    v[32] = fp32_to_intermed25(c);
    return accumulate_fp32_trunc(v, 33);
}

static fp16_t bw_dp32a_e4m3_fp16_group(const e4m3_t a[32], const e4m3_t b[32], fp16_t c) {
    for (int i = 0; i < 32; ++i) {
        if (e4m3_is_nan(a[i]) || e4m3_is_nan(b[i]))        return NAN_OUT_FP16;
    }
    if (fp16_is_nan(c)) return NAN_OUT_FP16;

    IntermVal v[33];
    for (int i = 0; i < 32; ++i) v[i] = e4m3_mul_fp16acc_25(a[i], b[i]);
    v[32] = fp16_to_intermed25(c);
    return accumulate_fp16_rne(v, 33);
}

static fp32_t bw_dp64a_e2m1_fp32_group(const e2m1_t a[64], const e2m1_t b[64], fp32_t c) {
    if (fp32_is_nan(c)) return NAN_OUT_FP32;

    IntermVal v[65];
    for (int i = 0; i < 64; ++i) v[i] = e2m1_mul_fp32acc_25(a[i], b[i]);
    v[64] = fp32_to_intermed25(c);
    return accumulate_fp32_trunc(v, 65);
}

static fp16_t bw_dp64a_e2m1_fp16_group(const e2m1_t a[64], const e2m1_t b[64], fp16_t c) {
    if (fp16_is_nan(c)) return NAN_OUT_FP16;

    IntermVal v[65];
    for (int i = 0; i < 64; ++i) v[i] = e2m1_mul_fp16acc_25(a[i], b[i]);
    v[64] = fp16_to_intermed25(c);
    return accumulate_fp16_rne(v, 65);
}

fp32_t blackwell_dp16a_fp32(const fp16_t* a, const fp16_t* b, size_t len, fp32_t c) {
    static constexpr fp16_t Z = 0x0000u;
    fp32_t acc = c;
    size_t i = 0;
    for (; i + 16 <= len; i += 16)
        acc = bw_dp16a_fp32_group(a + i, b + i, acc);
    if (i < len) {
        fp16_t pa[16] = {Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z};
        fp16_t pb[16] = {Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        acc = bw_dp16a_fp32_group(pa, pb, acc);
    }
    return acc;
}

fp16_t blackwell_dp16a_fp16(const fp16_t* a, const fp16_t* b, size_t len, fp16_t c) {
    static constexpr fp16_t Z = 0x0000u;
    fp16_t acc = c;
    size_t i = 0;
    for (; i + 16 <= len; i += 16)
        acc = bw_dp16a_fp16_group(a + i, b + i, acc);
    if (i < len) {
        fp16_t pa[16] = {Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z};
        fp16_t pb[16] = {Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        acc = bw_dp16a_fp16_group(pa, pb, acc);
    }
    return acc;
}

fp32_t blackwell_dp16a_bf16(const bf16_t* a, const bf16_t* b, size_t len, fp32_t c) {
    static constexpr bf16_t Z = 0x0000u;
    fp32_t acc = c;
    size_t i = 0;
    for (; i + 16 <= len; i += 16)
        acc = bw_dp16a_bf16_group(a + i, b + i, acc);
    if (i < len) {
        bf16_t pa[16] = {Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z};
        bf16_t pb[16] = {Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        acc = bw_dp16a_bf16_group(pa, pb, acc);
    }
    return acc;
}

fp32_t blackwell_dp32a_e5m2_fp32(const e5m2_t* a, const e5m2_t* b, size_t len, fp32_t c) {
    fp32_t acc = c;
    size_t i = 0;
    for (; i + 32 <= len; i += 32)
        acc = bw_dp32a_e5m2_fp32_group(a + i, b + i, acc);
    if (i < len) {
        e5m2_t pa[32] = {}; e5m2_t pb[32] = {};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        acc = bw_dp32a_e5m2_fp32_group(pa, pb, acc);
    }
    return acc;
}

fp16_t blackwell_dp32a_e5m2_fp16(const e5m2_t* a, const e5m2_t* b, size_t len, fp16_t c) {
    fp16_t acc = c;
    size_t i = 0;
    for (; i + 32 <= len; i += 32)
        acc = bw_dp32a_e5m2_fp16_group(a + i, b + i, acc);
    if (i < len) {
        e5m2_t pa[32] = {}; e5m2_t pb[32] = {};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        acc = bw_dp32a_e5m2_fp16_group(pa, pb, acc);
    }
    return acc;
}

fp32_t blackwell_dp32a_e4m3_fp32(const e4m3_t* a, const e4m3_t* b, size_t len, fp32_t c) {
    fp32_t acc = c;
    size_t i = 0;
    for (; i + 32 <= len; i += 32)
        acc = bw_dp32a_e4m3_fp32_group(a + i, b + i, acc);
    if (i < len) {
        e4m3_t pa[32] = {}; e4m3_t pb[32] = {};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        acc = bw_dp32a_e4m3_fp32_group(pa, pb, acc);
    }
    return acc;
}

fp16_t blackwell_dp32a_e4m3_fp16(const e4m3_t* a, const e4m3_t* b, size_t len, fp16_t c) {
    fp16_t acc = c;
    size_t i = 0;
    for (; i + 32 <= len; i += 32)
        acc = bw_dp32a_e4m3_fp16_group(a + i, b + i, acc);
    if (i < len) {
        e4m3_t pa[32] = {}; e4m3_t pb[32] = {};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        acc = bw_dp32a_e4m3_fp16_group(pa, pb, acc);
    }
    return acc;
}

fp32_t blackwell_dp64a_e2m1_fp32(const e2m1_t* a, const e2m1_t* b, size_t len, fp32_t c) {
    fp32_t acc = c;
    size_t i = 0;
    for (; i + 64 <= len; i += 64)
        acc = bw_dp64a_e2m1_fp32_group(a + i, b + i, acc);
    if (i < len) {
        e2m1_t pa[64] = {}; e2m1_t pb[64] = {};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        acc = bw_dp64a_e2m1_fp32_group(pa, pb, acc);
    }
    return acc;
}

fp16_t blackwell_dp64a_e2m1_fp16(const e2m1_t* a, const e2m1_t* b, size_t len, fp16_t c) {
    fp16_t acc = c;
    size_t i = 0;
    for (; i + 64 <= len; i += 64)
        acc = bw_dp64a_e2m1_fp16_group(a + i, b + i, acc);
    if (i < len) {
        e2m1_t pa[64] = {}; e2m1_t pb[64] = {};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        acc = bw_dp64a_e2m1_fp16_group(pa, pb, acc);
    }
    return acc;
}

static constexpr fp32_t FP32_POS_INF  = 0x7F800000u;
static constexpr fp32_t FP32_NEG_INF  = 0xFF800000u;

struct Interm35 {
    enum Kind { ZERO, NORMAL, INF, NAN_VAL } kind = ZERO;
    bool     sign = false;
    int32_t  E    = 0;
    uint64_t M    = 0;
};

static void decode_scale_e8(uint8_t val,
                             bool& is_nan, bool& is_zero,
                             uint32_t& Ma, int& Ea)
{
    is_nan  = (val == 0xFF);
    is_zero = (val == 0x00);
    Ma = 1;
    Ea = (int)val - 127;
}

static void decode_scale_ue4m3(uint8_t val,
                                bool& is_nan, bool& is_zero,
                                uint32_t& Ma, int& Ea)
{
    is_nan  = (val == 0x7F || val == 0xFF);
    is_zero = ((val & 0x7F) == 0);
    uint8_t body = val & 0x7F;
    int exp_f  = body >> 3;
    int mant_f = body & 0x7;
    if (exp_f == 0) { Ma = mant_f; Ea = -9; }
    else            { Ma = 8 | mant_f; Ea = exp_f - 10; }
}

static Interm35 fp32_to_interm35(fp32_t c)
{
    Interm35 r{};
    r.sign = fp32_sign(c);
    int      E  = fp32_exp(c);
    uint32_t Mf = fp32_mant(c);
    if (E == 255) {
        r.kind = (Mf == 0) ? Interm35::INF : Interm35::NAN_VAL;
        return r;
    }
    if (E == 0) {
        if (Mf == 0) { r.kind = Interm35::ZERO; return r; }
        r.kind = Interm35::NORMAL;
        r.E    = 0;
        r.M    = (uint64_t)Mf << 12;
        return r;
    }
    r.kind = Interm35::NORMAL;
    r.E    = E;
    r.M    = ((uint64_t)1 << 34) | ((uint64_t)Mf << 11);
    return r;
}

static Interm35 block_to_interm35(const e2m1_t* a, const e2m1_t* b,
                                   size_t N,
                                   uint8_t sa, uint8_t sb,
                                   bool is_e8)
{
    Interm35 r{};

    int64_t sum_P = 0;
    for (size_t i = 0; i < N; ++i) {
        if (e2m1_is_zero(a[i]) || e2m1_is_zero(b[i])) continue;
        E2M1Norm na = decode_e2m1(a[i]);
        E2M1Norm nb = decode_e2m1(b[i]);
        int exp_sum = na.exp + nb.exp;
        int shift   = exp_sum + 4;
        int64_t term = (int64_t)(na.sig * nb.sig) << shift;
        if (na.sign != nb.sign) sum_P -= term;
        else                    sum_P += term;
    }
    if (sum_P == 0) { r.kind = Interm35::ZERO; return r; }

    bool     is_nan_a, is_zero_a, is_nan_b, is_zero_b;
    uint32_t Ma, Mb; int Ea, Eb;
    if (is_e8) {
        decode_scale_e8   (sa, is_nan_a, is_zero_a, Ma, Ea);
        decode_scale_e8   (sb, is_nan_b, is_zero_b, Mb, Eb);
    } else {
        decode_scale_ue4m3(sa, is_nan_a, is_zero_a, Ma, Ea);
        decode_scale_ue4m3(sb, is_nan_b, is_zero_b, Mb, Eb);
    }
    if (is_nan_a || is_nan_b)   { r.kind = Interm35::NAN_VAL; return r; }
    if (is_zero_a || is_zero_b) { r.kind = Interm35::ZERO;    return r; }

    bool     blk_sign  = (sum_P < 0);
    uint64_t abs_P     = blk_sign ? (uint64_t)(-sum_P) : (uint64_t)sum_P;
    uint64_t sig_total = abs_P * (uint64_t)Ma * (uint64_t)Mb;
    if (sig_total == 0) { r.kind = Interm35::ZERO; return r; }

    int exp_total = Ea + Eb - 6;

    int lead  = 63 - __builtin_clzll(sig_total);
    int sft   = 34 - lead;
    uint64_t M35 = (sft >= 0) ? (sig_total << sft) : (sig_total >> (-sft));
    int E35 = exp_total + lead + 127;

    r.sign = blk_sign;
    if (E35 >= 255) {
        r.kind = Interm35::INF;
    } else {
        r.kind = Interm35::NORMAL;
        r.E    = E35;
        r.M    = M35;
    }
    return r;
}

static fp32_t accumulate_interm35_fp32(const Interm35* vals, int K)
{
    bool has_nan = false, has_pos_inf = false, has_neg_inf = false;
    for (int i = 0; i < K; ++i) {
        if (vals[i].kind == Interm35::NAN_VAL) { has_nan = true; continue; }
        if (vals[i].kind == Interm35::INF) {
            if (vals[i].sign) has_neg_inf = true;
            else              has_pos_inf = true;
        }
    }
    if (has_nan || (has_pos_inf && has_neg_inf)) return NAN_OUT_FP32;
    if (has_pos_inf) return FP32_POS_INF;
    if (has_neg_inf) return FP32_NEG_INF;

    int max_E = INT32_MIN;
    for (int i = 0; i < K; ++i)
        if (vals[i].kind == Interm35::NORMAL && vals[i].E > max_E)
            max_E = vals[i].E;
    if (max_E == INT32_MIN) return 0u;
    int64_t accum_M = 0;
    for (int i = 0; i < K; ++i) {
        if (vals[i].kind != Interm35::NORMAL) continue;
        int      shift      = max_E - vals[i].E;
        uint64_t aligned_M  = (shift < 64) ? (vals[i].M >> shift) : 0;
        if (vals[i].sign) accum_M -= (int64_t)aligned_M;
        else              accum_M += (int64_t)aligned_M;
    }
    if (accum_M == 0) return 0u;

    bool     final_sign = (accum_M < 0);
    uint64_t final_abs  = final_sign ? (uint64_t)(-accum_M) : (uint64_t)accum_M;
    int     lead     = 63 - __builtin_clzll(final_abs);
    int32_t E_out   = max_E + lead - 34;

    uint32_t mant23;
    if      (lead >= 23) mant23 = (uint32_t)(final_abs >> (lead - 23)) & 0x7FFFFFu;
    else                 mant23 = (uint32_t)(final_abs << (23 - lead)) & 0x7FFFFFu;

    uint32_t sign_bit = final_sign ? 0x80000000u : 0u;

    if (E_out >= 255) return sign_bit | 0x7F800000u;
    if (E_out <= 0) {
        int shift2 = 1 - E_out;
        if (shift2 > 24) return sign_bit;
        uint32_t sub_sig  = (1u << 23) | mant23;
        uint32_t sub_mant = sub_sig >> shift2;
        return sign_bit | sub_mant;
    }
    return sign_bit | ((uint32_t)E_out << 23) | mant23;
}

fp32_t blackwell_mxfp4_e2m1_e8_fp32(const e2m1_t*  a,  const e2m1_t*  b,
                                      const uint8_t* sa, const uint8_t* sb,
                                      size_t len, fp32_t c)
{
    fp32_t acc = c;
    size_t i = 0, g = 0;
    for (; i + 64 <= len; i += 64, g += 2) {
        Interm35 vals[3];
        vals[0] = block_to_interm35(a+i,    b+i,    32, sa[g],   sb[g],   true);
        vals[1] = block_to_interm35(a+i+32, b+i+32, 32, sa[g+1], sb[g+1], true);
        vals[2] = fp32_to_interm35(acc);
        acc = accumulate_interm35_fp32(vals, 3);
    }
    if (i < len) {
        e2m1_t pa[64] = {}, pb[64] = {};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        Interm35 vals[3];
        vals[0] = block_to_interm35(pa,    pb,    32, sa[g],   sb[g],   true);
        vals[1] = block_to_interm35(pa+32, pb+32, 32, sa[g+1], sb[g+1], true);
        vals[2] = fp32_to_interm35(acc);
        acc = accumulate_interm35_fp32(vals, 3);
    }
    return acc;
}

fp32_t blackwell_nvfp4_e2m1_ue4m3_fp32(const e2m1_t*  a,  const e2m1_t*  b,
                                         const uint8_t* sa, const uint8_t* sb,
                                         size_t len, fp32_t c)
{
    fp32_t acc = c;
    size_t i = 0, g = 0;
    for (; i + 64 <= len; i += 64, g += 4) {
        Interm35 vals[5];
        vals[0] = block_to_interm35(a+i,    b+i,    16, sa[g],   sb[g],   false);
        vals[1] = block_to_interm35(a+i+16, b+i+16, 16, sa[g+1], sb[g+1], false);
        vals[2] = block_to_interm35(a+i+32, b+i+32, 16, sa[g+2], sb[g+2], false);
        vals[3] = block_to_interm35(a+i+48, b+i+48, 16, sa[g+3], sb[g+3], false);
        vals[4] = fp32_to_interm35(acc);
        acc = accumulate_interm35_fp32(vals, 5);
    }
    if (i < len) {
        e2m1_t pa[64] = {}, pb[64] = {};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        Interm35 vals[5];
        vals[0] = block_to_interm35(pa,    pb,    16, sa[g],   sb[g],   false);
        vals[1] = block_to_interm35(pa+16, pb+16, 16, sa[g+1], sb[g+1], false);
        vals[2] = block_to_interm35(pa+32, pb+32, 16, sa[g+2], sb[g+2], false);
        vals[3] = block_to_interm35(pa+48, pb+48, 16, sa[g+3], sb[g+3], false);
        vals[4] = fp32_to_interm35(acc);
        acc = accumulate_interm35_fp32(vals, 5);
    }
    return acc;
}