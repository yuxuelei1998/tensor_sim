#include "hopper_tensor.h"
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

static void mul3_13(uint32_t sig_a, int32_t exp_a,
                    uint32_t sig_b, int32_t exp_b,
                    int32_t& out_exp, uint32_t& out_mant) {
    uint32_t prod = sig_a * sig_b;
    if (prod >= (1u << 5)) {

        out_mant = (prod - (1u << 5)) << 8;
        out_exp  = exp_a + exp_b + 1;
    } else {

        out_mant = (prod - (1u << 4)) << 9;
        out_exp  = exp_a + exp_b;
    }
}

static void mul4_13(uint32_t sig_a, int32_t exp_a,
                    uint32_t sig_b, int32_t exp_b,
                    int32_t& out_exp, uint32_t& out_mant) {
    uint32_t prod = sig_a * sig_b;
    if (prod >= (1u << 7)) {

        out_mant = (prod - (1u << 7)) << 6;
        out_exp  = exp_a + exp_b + 1;
    } else {

        out_mant = (prod - (1u << 6)) << 7;
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

static IntermVal e5m2_mul_fp32acc_13(e5m2_t a, e5m2_t b) {
    bool rs = e5m2_sign(a) ^ e5m2_sign(b);
    if (e5m2_is_nan(a) || e5m2_is_nan(b))                    return {IntermVal::NAN_VAL};
    if ((e5m2_is_zero(a) && e5m2_is_inf(b)) ||
        (e5m2_is_inf(a)  && e5m2_is_zero(b)))                return {IntermVal::NAN_VAL};
    if (e5m2_is_zero(a) || e5m2_is_zero(b))                  return {IntermVal::ZERO};
    if (e5m2_is_inf(a)  || e5m2_is_inf(b))
        return {rs ? IntermVal::NEG_INF : IntermVal::POS_INF, rs};
    E5M2Norm na = decode_e5m2(a), nb = decode_e5m2(b);
    int32_t e; uint32_t m;
    mul3_13(na.sig, na.exp, nb.sig, nb.exp, e, m);
    return {IntermVal::NORMAL, rs, e, m};
}

static IntermVal e5m2_mul_fp16acc_13(e5m2_t a, e5m2_t b) {
    bool rs = e5m2_sign(a) ^ e5m2_sign(b);
    if (e5m2_is_nan(a) || e5m2_is_nan(b))                    return {IntermVal::NAN_VAL};
    if ((e5m2_is_zero(a) && e5m2_is_inf(b)) ||
        (e5m2_is_inf(a)  && e5m2_is_zero(b)))                return {IntermVal::NAN_VAL};
    if (e5m2_is_zero(a) || e5m2_is_zero(b))                  return {IntermVal::ZERO};
    if (e5m2_is_inf(a)  || e5m2_is_inf(b))
        return {rs ? IntermVal::NEG_INF : IntermVal::POS_INF, rs};
    E5M2Norm na = decode_e5m2(a), nb = decode_e5m2(b);
    int32_t e; uint32_t m;
    mul3_13(na.sig, na.exp, nb.sig, nb.exp, e, m);
    if (e > 15)  return {rs ? IntermVal::NEG_INF_OVF : IntermVal::POS_INF_OVF, rs};
    if (e >= -14) return {IntermVal::NORMAL, rs, e, m};

    int shift = -14 - e;
    static constexpr int SHIFT_CAP = 14;
    if (shift >= SHIFT_CAP) return {IntermVal::ZERO};
    uint32_t full_sig = (1u << 13) | m;
    uint32_t sub_sig  = full_sig >> shift;
    if (sub_sig == 0) return {IntermVal::ZERO};

    int lead = 31 - __builtin_clz(sub_sig);
    int32_t new_exp = lead - 27;
    uint32_t new_mant;
    if (lead >= 13) new_mant = (sub_sig >> (lead - 13)) & 0x1FFFu;
    else            new_mant = (sub_sig << (13 - lead)) & 0x1FFFu;
    return {IntermVal::NORMAL, rs, new_exp, new_mant};
}

static IntermVal e4m3_mul_fp32acc_13(e4m3_t a, e4m3_t b) {
    bool rs = e4m3_sign(a) ^ e4m3_sign(b);
    if (e4m3_is_nan(a) || e4m3_is_nan(b))                    return {IntermVal::NAN_VAL};

    if (e4m3_is_zero(a) || e4m3_is_zero(b))                  return {IntermVal::ZERO};
    E4M3Norm na = decode_e4m3(a), nb = decode_e4m3(b);
    int32_t e; uint32_t m;
    mul4_13(na.sig, na.exp, nb.sig, nb.exp, e, m);
    return {IntermVal::NORMAL, rs, e, m};
}

static IntermVal e4m3_mul_fp16acc_13(e4m3_t a, e4m3_t b) {
    bool rs = e4m3_sign(a) ^ e4m3_sign(b);
    if (e4m3_is_nan(a) || e4m3_is_nan(b))                    return {IntermVal::NAN_VAL};
    if (e4m3_is_zero(a) || e4m3_is_zero(b))                  return {IntermVal::ZERO};
    E4M3Norm na = decode_e4m3(a), nb = decode_e4m3(b);
    int32_t e; uint32_t m;
    mul4_13(na.sig, na.exp, nb.sig, nb.exp, e, m);
    if (e > 15)  return {rs ? IntermVal::NEG_INF_OVF : IntermVal::POS_INF_OVF, rs};
    if (e >= -14) return {IntermVal::NORMAL, rs, e, m};

    int shift = -14 - e;
    static constexpr int SHIFT_CAP = 14;
    if (shift >= SHIFT_CAP) return {IntermVal::ZERO};
    uint32_t full_sig = (1u << 13) | m;
    uint32_t sub_sig  = full_sig >> shift;
    if (sub_sig == 0) return {IntermVal::ZERO};
    int lead = 31 - __builtin_clz(sub_sig);
    int32_t new_exp = lead - 27;
    uint32_t new_mant;
    if (lead >= 13) new_mant = (sub_sig >> (lead - 13)) & 0x1FFFu;
    else            new_mant = (sub_sig << (13 - lead)) & 0x1FFFu;
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
        int32_t  unb  = -127 - lz;
        int frac_shift = 25 - (22 - lz);
        uint32_t mant25 = (m - (1u << (22 - lz))) << frac_shift;
        return {IntermVal::NORMAL, sc, unb, mant25};
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

static IntermVal fp32_to_intermed13(fp32_t c) {
    bool sc = fp32_sign(c);
    int  e  = fp32_exp(c);
    uint32_t m = fp32_mant(c);
    if (e == 0xFF && m != 0) return {IntermVal::NAN_VAL};
    if (e == 0xFF && m == 0) return {sc ? IntermVal::NEG_INF : IntermVal::POS_INF, sc};
    if (fp32_is_zero(c))     return {IntermVal::ZERO};
    if (e == 0) {

        return {IntermVal::ZERO};
    }

    return {IntermVal::NORMAL, sc, e - 127, (m >> 10) & 0x1FFFu};
}

static IntermVal fp16_to_intermed13(fp16_t c) {
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

        uint32_t mant13 = (sig11 & 0x3FFu) << 3;
        return {IntermVal::NORMAL, sc, unb, mant13};
    }

    return {IntermVal::NORMAL, sc, e - 15, (m << 3) & 0x1FFFu};
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

static void align_accum_n(const IntermVal* v, int n, int mant_bits,
                           bool& sign_out, uint64_t& abs_out, int32_t& max_exp_out) {
    int32_t max_exp = INT32_MIN;
    for (int i = 0; i < n; ++i)
        if (v[i].kind == IntermVal::NORMAL)
            max_exp = std::max(max_exp, v[i].exp);
    max_exp_out = max_exp;

    if (max_exp == INT32_MIN) { sign_out = false; abs_out = 0; return; }

    const int shift_cap = mant_bits + 1;
    int64_t accum = 0;
    for (int i = 0; i < n; ++i) {
        if (v[i].kind != IntermVal::NORMAL) continue;
        int shift = max_exp - v[i].exp;
        if (shift >= shift_cap) continue;
        uint64_t sig     = ((uint64_t)1 << mant_bits) | v[i].mant;
        uint64_t aligned = sig >> shift;
        if (v[i].sign) accum -= (int64_t)aligned;
        else           accum += (int64_t)aligned;
    }

    sign_out = accum < 0;
    abs_out  = sign_out ? (uint64_t)(-accum) : (uint64_t)accum;
}

static fp32_t accumulate_fp32_trunc(const IntermVal* v, int n, int mant_bits) {
    fp32_t sig_out;
    if (check_specials_n<fp32_t>(v, n, NAN_OUT_FP32, 0x7F800000u, 0xFF800000u, sig_out))
        return sig_out;

    bool rs; uint64_t abs_sum; int32_t max_exp;
    align_accum_n(v, n, mant_bits, rs, abs_sum, max_exp);
    if (abs_sum == 0) return 0u;

    int lead    = 63 - __builtin_clzll(abs_sum);
    int32_t unb = lead + max_exp - mant_bits;
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

static fp16_t accumulate_fp16_rne(const IntermVal* v, int n, int mant_bits) {
    fp16_t sig_out;
    if (check_specials_n<fp16_t>(v, n, NAN_OUT_FP16, 0x7C00u, 0xFC00u, sig_out))
        return sig_out;

    bool rs; uint64_t abs_sum; int32_t max_exp;
    align_accum_n(v, n, mant_bits, rs, abs_sum, max_exp);
    if (abs_sum == 0) return 0u;

    int lead    = 63 - __builtin_clzll(abs_sum);
    int32_t unb = lead + max_exp - mant_bits;

    uint32_t mantX; bool sticky = false;
    if (lead >= mant_bits) {
        int rs2 = lead - mant_bits;
        mantX   = (uint32_t)((abs_sum >> rs2) & ((1u << mant_bits) - 1));
        sticky  = (rs2 > 0) && ((abs_sum & ((1ULL << rs2) - 1)) != 0);
    } else {
        mantX = (uint32_t)((abs_sum << (mant_bits - lead)) & ((1u << mant_bits) - 1));
    }

    const int RND_SHIFT = mant_bits - 10;

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

        int shift_s = mant_bits - 24 - unb;
        uint64_t sig_full = ((uint64_t)1 << mant_bits) | mantX;

        if (shift_s > mant_bits) {

            int rnd_pos = shift_s - 1;
            int rnd_bit = (rnd_pos <= mant_bits) ? (int)((sig_full >> rnd_pos) & 1) : 0;
            bool stk2   = sticky || (rnd_pos > 0 && rnd_pos <= mant_bits &&
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

static fp32_t hopper_dp16a_fp32_group(const fp16_t a[16], const fp16_t b[16], fp32_t c) {
    for (int i = 0; i < 16; ++i) {
        if (fp16_is_nan(a[i]) || fp16_is_nan(b[i]))        return NAN_OUT_FP32;
        if (fp16_is_zero(a[i]) && fp16_is_inf(b[i]))       return NAN_OUT_FP32;
        if (fp16_is_inf(a[i])  && fp16_is_zero(b[i]))      return NAN_OUT_FP32;
    }
    if (fp32_is_nan(c)) return NAN_OUT_FP32;

    IntermVal v[17];
    for (int i = 0; i < 16; ++i) v[i] = fp16_mul_fp32acc_25(a[i], b[i]);
    v[16] = fp32_to_intermed25(c);
    return accumulate_fp32_trunc(v, 17, 25);
}

static fp16_t hopper_dp16a_fp16_group(const fp16_t a[16], const fp16_t b[16], fp16_t c) {
    for (int i = 0; i < 16; ++i) {
        if (fp16_is_nan(a[i]) || fp16_is_nan(b[i]))        return NAN_OUT_FP16;
        if (fp16_is_zero(a[i]) && fp16_is_inf(b[i]))       return NAN_OUT_FP16;
        if (fp16_is_inf(a[i])  && fp16_is_zero(b[i]))      return NAN_OUT_FP16;
    }
    if (fp16_is_nan(c)) return NAN_OUT_FP16;

    IntermVal v[17];
    for (int i = 0; i < 16; ++i) v[i] = fp16_mul_fp16acc_25(a[i], b[i]);
    v[16] = fp16_to_intermed25(c);
    return accumulate_fp16_rne(v, 17, 25);
}

static fp32_t hopper_dp16a_bf16_group(const bf16_t a[16], const bf16_t b[16], fp32_t c) {
    for (int i = 0; i < 16; ++i) {
        if (bf16_is_nan(a[i]) || bf16_is_nan(b[i]))        return NAN_OUT_FP32;
        if (bf16_is_zero(a[i]) && bf16_is_inf(b[i]))       return NAN_OUT_FP32;
        if (bf16_is_inf(a[i])  && bf16_is_zero(b[i]))      return NAN_OUT_FP32;
    }
    if (fp32_is_nan(c)) return NAN_OUT_FP32;

    IntermVal v[17];
    for (int i = 0; i < 16; ++i) v[i] = bf16_mul_fp32acc_25(a[i], b[i]);
    v[16] = fp32_to_intermed25(c);
    return accumulate_fp32_trunc(v, 17, 25);
}

static fp32_t hopper_dp32a_e5m2_fp32_group(const e5m2_t a[32], const e5m2_t b[32], fp32_t c) {
    for (int i = 0; i < 32; ++i) {
        if (e5m2_is_nan(a[i]) || e5m2_is_nan(b[i]))        return NAN_OUT_FP32;
        if (e5m2_is_zero(a[i]) && e5m2_is_inf(b[i]))       return NAN_OUT_FP32;
        if (e5m2_is_inf(a[i])  && e5m2_is_zero(b[i]))      return NAN_OUT_FP32;
    }
    if (fp32_is_nan(c)) return NAN_OUT_FP32;

    IntermVal v[33];
    for (int i = 0; i < 32; ++i) v[i] = e5m2_mul_fp32acc_13(a[i], b[i]);
    v[32] = fp32_to_intermed13(c);
    return accumulate_fp32_trunc(v, 33, 13);
}

static fp16_t hopper_dp32a_e5m2_fp16_group(const e5m2_t a[32], const e5m2_t b[32], fp16_t c) {
    for (int i = 0; i < 32; ++i) {
        if (e5m2_is_nan(a[i]) || e5m2_is_nan(b[i]))        return NAN_OUT_FP16;
        if (e5m2_is_zero(a[i]) && e5m2_is_inf(b[i]))       return NAN_OUT_FP16;
        if (e5m2_is_inf(a[i])  && e5m2_is_zero(b[i]))      return NAN_OUT_FP16;
    }
    if (fp16_is_nan(c)) return NAN_OUT_FP16;

    IntermVal v[33];
    for (int i = 0; i < 32; ++i) v[i] = e5m2_mul_fp16acc_13(a[i], b[i]);
    v[32] = fp16_to_intermed13(c);
    return accumulate_fp16_rne(v, 33, 13);
}

static fp32_t hopper_dp32a_e4m3_fp32_group(const e4m3_t a[32], const e4m3_t b[32], fp32_t c) {
    for (int i = 0; i < 32; ++i) {
        if (e4m3_is_nan(a[i]) || e4m3_is_nan(b[i]))        return NAN_OUT_FP32;

    }
    if (fp32_is_nan(c)) return NAN_OUT_FP32;

    IntermVal v[33];
    for (int i = 0; i < 32; ++i) v[i] = e4m3_mul_fp32acc_13(a[i], b[i]);
    v[32] = fp32_to_intermed13(c);
    return accumulate_fp32_trunc(v, 33, 13);
}

static fp16_t hopper_dp32a_e4m3_fp16_group(const e4m3_t a[32], const e4m3_t b[32], fp16_t c) {
    for (int i = 0; i < 32; ++i) {
        if (e4m3_is_nan(a[i]) || e4m3_is_nan(b[i]))        return NAN_OUT_FP16;
    }
    if (fp16_is_nan(c)) return NAN_OUT_FP16;

    IntermVal v[33];
    for (int i = 0; i < 32; ++i) v[i] = e4m3_mul_fp16acc_13(a[i], b[i]);
    v[32] = fp16_to_intermed13(c);
    return accumulate_fp16_rne(v, 33, 13);
}

fp32_t hopper_dp16a_fp32(const fp16_t* a, const fp16_t* b, size_t len, fp32_t c) {
    static constexpr fp16_t Z = 0x0000u;
    fp32_t acc = c;
    size_t i = 0;
    for (; i + 16 <= len; i += 16)
        acc = hopper_dp16a_fp32_group(a + i, b + i, acc);
    if (i < len) {
        fp16_t pa[16] = {Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z};
        fp16_t pb[16] = {Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        acc = hopper_dp16a_fp32_group(pa, pb, acc);
    }
    return acc;
}

fp16_t hopper_dp16a_fp16(const fp16_t* a, const fp16_t* b, size_t len, fp16_t c) {
    static constexpr fp16_t Z = 0x0000u;
    fp16_t acc = c;
    size_t i = 0;
    for (; i + 16 <= len; i += 16)
        acc = hopper_dp16a_fp16_group(a + i, b + i, acc);
    if (i < len) {
        fp16_t pa[16] = {Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z};
        fp16_t pb[16] = {Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        acc = hopper_dp16a_fp16_group(pa, pb, acc);
    }
    return acc;
}

fp32_t hopper_dp16a_bf16(const bf16_t* a, const bf16_t* b, size_t len, fp32_t c) {
    static constexpr bf16_t Z = 0x0000u;
    fp32_t acc = c;
    size_t i = 0;
    for (; i + 16 <= len; i += 16)
        acc = hopper_dp16a_bf16_group(a + i, b + i, acc);
    if (i < len) {
        bf16_t pa[16] = {Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z};
        bf16_t pb[16] = {Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z,Z};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        acc = hopper_dp16a_bf16_group(pa, pb, acc);
    }
    return acc;
}

fp32_t hopper_dp32a_e5m2_fp32(const e5m2_t* a, const e5m2_t* b, size_t len, fp32_t c) {
    fp32_t acc = c;
    size_t i = 0;
    for (; i + 32 <= len; i += 32)
        acc = hopper_dp32a_e5m2_fp32_group(a + i, b + i, acc);
    if (i < len) {
        e5m2_t pa[32] = {}; e5m2_t pb[32] = {};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        acc = hopper_dp32a_e5m2_fp32_group(pa, pb, acc);
    }
    return acc;
}

fp16_t hopper_dp32a_e5m2_fp16(const e5m2_t* a, const e5m2_t* b, size_t len, fp16_t c) {
    fp16_t acc = c;
    size_t i = 0;
    for (; i + 32 <= len; i += 32)
        acc = hopper_dp32a_e5m2_fp16_group(a + i, b + i, acc);
    if (i < len) {
        e5m2_t pa[32] = {}; e5m2_t pb[32] = {};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        acc = hopper_dp32a_e5m2_fp16_group(pa, pb, acc);
    }
    return acc;
}

fp32_t hopper_dp32a_e4m3_fp32(const e4m3_t* a, const e4m3_t* b, size_t len, fp32_t c) {
    fp32_t acc = c;
    size_t i = 0;
    for (; i + 32 <= len; i += 32)
        acc = hopper_dp32a_e4m3_fp32_group(a + i, b + i, acc);
    if (i < len) {
        e4m3_t pa[32] = {}; e4m3_t pb[32] = {};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        acc = hopper_dp32a_e4m3_fp32_group(pa, pb, acc);
    }
    return acc;
}

fp16_t hopper_dp32a_e4m3_fp16(const e4m3_t* a, const e4m3_t* b, size_t len, fp16_t c) {
    fp16_t acc = c;
    size_t i = 0;
    for (; i + 32 <= len; i += 32)
        acc = hopper_dp32a_e4m3_fp16_group(a + i, b + i, acc);
    if (i < len) {
        e4m3_t pa[32] = {}; e4m3_t pb[32] = {};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        acc = hopper_dp32a_e4m3_fp16_group(pa, pb, acc);
    }
    return acc;
}