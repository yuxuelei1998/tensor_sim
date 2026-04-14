#include "ampere_tensor.h"
#include <algorithm>
#include <climits>

static constexpr fp32_t NAN_OUT_FP32 = 0x7FFFFFFFu;
static constexpr fp16_t NAN_OUT_FP16 = 0x7FFFu;

struct IntermVal {
    enum Kind : uint8_t { ZERO, NORMAL, POS_INF, NEG_INF, NAN_VAL } kind = ZERO;
    bool     sign = false;
    int32_t  exp  = 0;
    uint32_t mant = 0;
};

struct FP16Norm { bool sign; int32_t exp; uint32_t sig; };

static FP16Norm decode_fp16_norm(fp16_t x) {
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

struct BF16Norm { bool sign; int32_t exp; uint32_t sig; };

static BF16Norm decode_bf16_norm(bf16_t x) {
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

static void mul11_24(uint32_t sig_a, int32_t exp_a,
                     uint32_t sig_b, int32_t exp_b,
                     int32_t& out_exp, uint32_t& out_mant24) {
    uint32_t prod = sig_a * sig_b;
    if (prod >= (1u << 21)) {

        out_mant24 = (prod - (1u << 21)) << 3;
        out_exp    = exp_a + exp_b + 1;
    } else {

        out_mant24 = (prod - (1u << 20)) << 4;
        out_exp    = exp_a + exp_b;
    }
}

static void mul8_24(uint32_t sig_a, int32_t exp_a,
                    uint32_t sig_b, int32_t exp_b,
                    int32_t& out_exp, uint32_t& out_mant24) {
    uint32_t prod = sig_a * sig_b;
    if (prod >= (1u << 15)) {

        out_mant24 = (prod - (1u << 15)) << 9;
        out_exp    = exp_a + exp_b + 1;
    } else {

        out_mant24 = (prod - (1u << 14)) << 10;
        out_exp    = exp_a + exp_b;
    }
}

static IntermVal fp16_mul_fp32_intermed(fp16_t a, fp16_t b) {
    bool rs = fp16_sign(a) ^ fp16_sign(b);
    if (fp16_is_nan(a) || fp16_is_nan(b))                    return {IntermVal::NAN_VAL};
    if ((fp16_is_zero(a) && fp16_is_inf(b)) ||
        (fp16_is_inf(a)  && fp16_is_zero(b)))                return {IntermVal::NAN_VAL};
    if (fp16_is_zero(a) || fp16_is_zero(b))                  return {IntermVal::ZERO};
    if (fp16_is_inf(a)  || fp16_is_inf(b))
        return {rs ? IntermVal::NEG_INF : IntermVal::POS_INF, rs};
    FP16Norm na = decode_fp16_norm(a);
    FP16Norm nb = decode_fp16_norm(b);
    int32_t e; uint32_t m;
    mul11_24(na.sig, na.exp, nb.sig, nb.exp, e, m);
    return {IntermVal::NORMAL, rs, e, m};
}

static IntermVal fp16_mul_fp16_intermed(fp16_t a, fp16_t b) {
    bool rs = fp16_sign(a) ^ fp16_sign(b);
    if (fp16_is_nan(a) || fp16_is_nan(b))                    return {IntermVal::NAN_VAL};
    if ((fp16_is_zero(a) && fp16_is_inf(b)) ||
        (fp16_is_inf(a)  && fp16_is_zero(b)))                return {IntermVal::NAN_VAL};
    if (fp16_is_zero(a) || fp16_is_zero(b))                  return {IntermVal::ZERO};
    if (fp16_is_inf(a)  || fp16_is_inf(b))
        return {rs ? IntermVal::NEG_INF : IntermVal::POS_INF, rs};

    FP16Norm na = decode_fp16_norm(a);
    FP16Norm nb = decode_fp16_norm(b);
    int32_t e; uint32_t m;
    mul11_24(na.sig, na.exp, nb.sig, nb.exp, e, m);

    if (e > 15) return {rs ? IntermVal::NEG_INF : IntermVal::POS_INF, rs};
    if (e >= -14) return {IntermVal::NORMAL, rs, e, m};

    int shift = -14 - e;

    if (shift >= 25) return {IntermVal::ZERO};
    uint32_t full_sig = (1u << 24) | m;
    uint32_t sub_sig  = full_sig >> shift;
    if (sub_sig == 0) return {IntermVal::ZERO};

    int lead = 31 - __builtin_clz(sub_sig);
    int32_t new_exp = lead - 38;
    uint32_t new_mant;
    if (lead >= 24) new_mant = (sub_sig >> (lead - 24)) & 0xFFFFFFu;
    else            new_mant = (sub_sig << (24 - lead)) & 0xFFFFFFu;
    return {IntermVal::NORMAL, rs, new_exp, new_mant};
}

static IntermVal bf16_mul_fp32_intermed(bf16_t a, bf16_t b) {
    bool rs = bf16_sign(a) ^ bf16_sign(b);
    if (bf16_is_nan(a) || bf16_is_nan(b))                    return {IntermVal::NAN_VAL};
    if ((bf16_is_zero(a) && bf16_is_inf(b)) ||
        (bf16_is_inf(a)  && bf16_is_zero(b)))                return {IntermVal::NAN_VAL};
    if (bf16_is_zero(a) || bf16_is_zero(b))                  return {IntermVal::ZERO};
    if (bf16_is_inf(a)  || bf16_is_inf(b))
        return {rs ? IntermVal::NEG_INF : IntermVal::POS_INF, rs};
    BF16Norm na = decode_bf16_norm(a);
    BF16Norm nb = decode_bf16_norm(b);
    int32_t e; uint32_t m;
    mul8_24(na.sig, na.exp, nb.sig, nb.exp, e, m);
    return {IntermVal::NORMAL, rs, e, m};
}

static IntermVal fp32_to_intermed(fp32_t c) {
    bool sc = fp32_sign(c);
    int  e  = fp32_exp(c);
    uint32_t m = fp32_mant(c);

    if (e == 0xFF && m != 0) return {IntermVal::NAN_VAL};
    if (e == 0xFF && m == 0) return {sc ? IntermVal::NEG_INF : IntermVal::POS_INF, sc};
    if (fp32_is_zero(c))     return {IntermVal::ZERO};

    if (e == 0) {

        int lz = __builtin_clz(m) - 9;
        int32_t  unb_exp  = -127 - lz;

        int frac_shift = 24 - (22 - lz);
        uint32_t mant24 = (m - (1u << (22 - lz))) << frac_shift;
        return {IntermVal::NORMAL, sc, unb_exp, mant24};
    }

    return {IntermVal::NORMAL, sc, e - 127, m << 1};
}

static IntermVal fp16_to_intermed(fp16_t c) {
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
        uint32_t mant24 = (sig11 & 0x3FFu) << 14;
        return {IntermVal::NORMAL, sc, unb, mant24};
    }

    return {IntermVal::NORMAL, sc, e - 15, m << 14};
}

template<typename T>
static bool check_specials9(const IntermVal v[9], T nan_out, T pos_inf_out, T neg_inf_out, T& out) {
    for (int i = 0; i < 9; ++i)
        if (v[i].kind == IntermVal::NAN_VAL) { out = nan_out; return true; }
    bool hp = false, hn = false;
    for (int i = 0; i < 9; ++i) {
        if (v[i].kind == IntermVal::POS_INF) hp = true;
        if (v[i].kind == IntermVal::NEG_INF) hn = true;
    }
    if (hp && hn) { out = nan_out;     return true; }
    if (hp)       { out = pos_inf_out; return true; }
    if (hn)       { out = neg_inf_out; return true; }
    return false;
}

static void align_accum9(const IntermVal v[9],
                         bool& sign_out, uint64_t& abs_out, int32_t& max_exp_out) {
    int32_t max_exp = INT32_MIN;
    for (int i = 0; i < 9; ++i)
        if (v[i].kind == IntermVal::NORMAL)
            max_exp = std::max(max_exp, v[i].exp);
    max_exp_out = max_exp;

    if (max_exp == INT32_MIN) { sign_out = false; abs_out = 0; return; }

    static constexpr int SHIFT_CAP = 25;
    int64_t accum = 0;
    for (int i = 0; i < 9; ++i) {
        if (v[i].kind != IntermVal::NORMAL) continue;
        int shift = max_exp - v[i].exp;
        if (shift >= SHIFT_CAP) continue;
        uint64_t sig     = ((uint64_t)1 << 24) | v[i].mant;
        uint64_t aligned = sig >> shift;
        if (v[i].sign) accum -= (int64_t)aligned;
        else           accum += (int64_t)aligned;
    }

    sign_out = accum < 0;
    abs_out  = sign_out ? (uint64_t)(-accum) : (uint64_t)accum;
}

static fp32_t accumulate_fp32(const IntermVal v[9]) {
    fp32_t sig_out;
    if (check_specials9<fp32_t>(v, NAN_OUT_FP32, 0x7F800000u, 0xFF800000u, sig_out))
        return sig_out;

    bool rs; uint64_t abs_sum; int32_t max_exp;
    align_accum9(v, rs, abs_sum, max_exp);
    if (abs_sum == 0) return 0u;

    int lead     = 63 - __builtin_clzll(abs_sum);
    int32_t unb  = lead + max_exp - 24;
    int32_t bsd  = unb + 127;

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

static fp16_t accumulate_fp16(const IntermVal v[9]) {
    fp16_t sig_out;
    if (check_specials9<fp16_t>(v, NAN_OUT_FP16, 0x7C00u, 0xFC00u, sig_out))
        return sig_out;

    bool rs; uint64_t abs_sum; int32_t max_exp;
    align_accum9(v, rs, abs_sum, max_exp);
    if (abs_sum == 0) return 0u;

    int lead    = 63 - __builtin_clzll(abs_sum);
    int32_t unb = lead + max_exp - 24;

    uint32_t mant24; bool sticky = false;
    if (lead >= 24) {
        int rs2 = lead - 24;
        mant24  = (uint32_t)((abs_sum >> rs2) & 0xFFFFFFu);
        sticky  = (rs2 > 0) && ((abs_sum & ((1ULL << rs2) - 1)) != 0);
    } else {
        mant24 = (uint32_t)((abs_sum << (24 - lead)) & 0xFFFFFFu);
    }

    auto rne_round_up = [](uint32_t mant, int round_b, bool stk) -> bool {
        if (!round_b) return false;
        if (stk)      return true;
        return (mant & 1u) != 0;
    };

    if (unb >= -14) {
        if (unb > 15) goto overflow_inf;

        uint32_t mant10  = mant24 >> 14;
        int      rnd_bit = (mant24 >> 13) & 1;
        bool     stk     = ((mant24 & 0x1FFFu) != 0) || sticky;
        if (rne_round_up(mant10, rnd_bit, stk)) {
            mant10++;
            if (mant10 >= (1u << 10)) { mant10 = 0; unb++; }
        }
        if (unb > 15) goto overflow_inf;
        uint32_t e16 = (uint32_t)(unb + 15);
        return (fp16_t)(((uint32_t)rs << 15) | (e16 << 10) | mant10);
    }

    {

        int ts = -unb;
        uint64_t sig25 = ((uint64_t)1 << 24) | mant24;

        if (ts > 24) {
            int rnd_pos = ts - 1;
            int rnd_bit = (rnd_pos < 25) ? (int)((sig25 >> rnd_pos) & 1) : 0;
            bool stk    = sticky || (rnd_pos > 0 && rnd_pos <= 24 &&
                          ((sig25 & ((1ULL << rnd_pos) - 1)) != 0));
            bool ru = rne_round_up(0u, rnd_bit, stk);
            if (!ru) return (fp16_t)((uint32_t)rs << 15);
            return (fp16_t)(((uint32_t)rs << 15) | 1u);
        }

        uint32_t mant10 = (uint32_t)(sig25 >> ts) & 0x3FFu;
        int rnd_bit = (ts > 0) ? (int)((sig25 >> (ts - 1)) & 1) : 0;
        bool stk = sticky ||
                   (ts > 1 && ((sig25 & ((1ULL << (ts - 1)) - 1)) != 0));
        if (rne_round_up(mant10, rnd_bit, stk)) {
            mant10++;
            if (mant10 >= (1u << 10)) {
                return (fp16_t)(((uint32_t)rs << 15) | (1u << 10));
            }
        }
        if (mant10 == 0) return (fp16_t)((uint32_t)rs << 15);
        return (fp16_t)(((uint32_t)rs << 15) | mant10);
    }

overflow_inf:
    return (fp16_t)(((uint32_t)rs << 15) | 0x7C00u);
}

static fp32_t ampere_dp8a_fp32_group(const fp16_t a[8], const fp16_t b[8], fp32_t c) {
    for (int i = 0; i < 8; ++i) {
        if (fp16_is_nan(a[i]) || fp16_is_nan(b[i]))          return NAN_OUT_FP32;
        if (fp16_is_zero(a[i]) && fp16_is_inf(b[i]))         return NAN_OUT_FP32;
        if (fp16_is_inf(a[i])  && fp16_is_zero(b[i]))        return NAN_OUT_FP32;
    }
    if (fp32_is_nan(c)) return NAN_OUT_FP32;

    IntermVal v[9];
    for (int i = 0; i < 8; ++i) v[i] = fp16_mul_fp32_intermed(a[i], b[i]);
    v[8] = fp32_to_intermed(c);
    return accumulate_fp32(v);
}

static fp16_t ampere_dp8a_fp16_group(const fp16_t a[8], const fp16_t b[8], fp16_t c) {
    for (int i = 0; i < 8; ++i) {
        if (fp16_is_nan(a[i]) || fp16_is_nan(b[i]))          return NAN_OUT_FP16;
        if (fp16_is_zero(a[i]) && fp16_is_inf(b[i]))         return NAN_OUT_FP16;
        if (fp16_is_inf(a[i])  && fp16_is_zero(b[i]))        return NAN_OUT_FP16;
    }
    if (fp16_is_nan(c)) return NAN_OUT_FP16;

    IntermVal v[9];
    for (int i = 0; i < 8; ++i) v[i] = fp16_mul_fp16_intermed(a[i], b[i]);
    v[8] = fp16_to_intermed(c);
    return accumulate_fp16(v);
}

static fp32_t ampere_dp8a_bf16_group(const bf16_t a[8], const bf16_t b[8], fp32_t c) {
    for (int i = 0; i < 8; ++i) {
        if (bf16_is_nan(a[i]) || bf16_is_nan(b[i]))          return NAN_OUT_FP32;
        if (bf16_is_zero(a[i]) && bf16_is_inf(b[i]))         return NAN_OUT_FP32;
        if (bf16_is_inf(a[i])  && bf16_is_zero(b[i]))        return NAN_OUT_FP32;
    }
    if (fp32_is_nan(c)) return NAN_OUT_FP32;

    IntermVal v[9];
    for (int i = 0; i < 8; ++i) v[i] = bf16_mul_fp32_intermed(a[i], b[i]);
    v[8] = fp32_to_intermed(c);
    return accumulate_fp32(v);
}

fp32_t ampere_dp8a_fp32(const fp16_t* a, const fp16_t* b, size_t len, fp32_t c) {
    static constexpr fp16_t ZERO16 = 0x0000u;
    fp32_t acc = c;
    size_t i = 0;
    for (; i + 8 <= len; i += 8)
        acc = ampere_dp8a_fp32_group(a + i, b + i, acc);
    if (i < len) {
        fp16_t pa[8] = {ZERO16,ZERO16,ZERO16,ZERO16,ZERO16,ZERO16,ZERO16,ZERO16};
        fp16_t pb[8] = {ZERO16,ZERO16,ZERO16,ZERO16,ZERO16,ZERO16,ZERO16,ZERO16};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        acc = ampere_dp8a_fp32_group(pa, pb, acc);
    }
    return acc;
}

fp16_t ampere_dp8a_fp16(const fp16_t* a, const fp16_t* b, size_t len, fp16_t c) {
    static constexpr fp16_t ZERO16 = 0x0000u;
    fp16_t acc = c;
    size_t i = 0;
    for (; i + 8 <= len; i += 8)
        acc = ampere_dp8a_fp16_group(a + i, b + i, acc);
    if (i < len) {
        fp16_t pa[8] = {ZERO16,ZERO16,ZERO16,ZERO16,ZERO16,ZERO16,ZERO16,ZERO16};
        fp16_t pb[8] = {ZERO16,ZERO16,ZERO16,ZERO16,ZERO16,ZERO16,ZERO16,ZERO16};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        acc = ampere_dp8a_fp16_group(pa, pb, acc);
    }
    return acc;
}

fp32_t ampere_dp8a_bf16(const bf16_t* a, const bf16_t* b, size_t len, fp32_t c) {
    static constexpr bf16_t ZERO_BF16 = 0x0000u;
    fp32_t acc = c;
    size_t i = 0;
    for (; i + 8 <= len; i += 8)
        acc = ampere_dp8a_bf16_group(a + i, b + i, acc);
    if (i < len) {
        bf16_t pa[8] = {ZERO_BF16,ZERO_BF16,ZERO_BF16,ZERO_BF16,
                        ZERO_BF16,ZERO_BF16,ZERO_BF16,ZERO_BF16};
        bf16_t pb[8] = {ZERO_BF16,ZERO_BF16,ZERO_BF16,ZERO_BF16,
                        ZERO_BF16,ZERO_BF16,ZERO_BF16,ZERO_BF16};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        acc = ampere_dp8a_bf16_group(pa, pb, acc);
    }
    return acc;
}