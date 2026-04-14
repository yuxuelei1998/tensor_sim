#include "volta_tensor.h"
#include <algorithm>
#include <climits>
#include <cassert>

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
        r.exp = -15 - lz;
        r.sig = m << (lz + 1);
    } else {
        r.exp = e - 15;
        r.sig = (1u << 10) | m;
    }
    return r;
}

static void mul11(uint32_t sig_a, int32_t exp_a,
                  uint32_t sig_b, int32_t exp_b,
                  int32_t& out_exp, uint32_t& out_mant23) {
    uint32_t prod = sig_a * sig_b;
    if (prod >= (1u << 21)) {

        out_mant23 = (prod - (1u << 21)) << 2;
        out_exp    = exp_a + exp_b + 1;
    } else {

        out_mant23 = (prod - (1u << 20)) << 3;
        out_exp    = exp_a + exp_b;
    }
}

static IntermVal fp16_mul_fp32_intermed(fp16_t a, fp16_t b) {
    bool rs = fp16_sign(a) ^ fp16_sign(b);

    if (fp16_is_nan(a) || fp16_is_nan(b))                              return {IntermVal::NAN_VAL};
    if ((fp16_is_zero(a) && fp16_is_inf(b)) ||
        (fp16_is_inf(a)  && fp16_is_zero(b)))                          return {IntermVal::NAN_VAL};
    if (fp16_is_zero(a) || fp16_is_zero(b))                            return {IntermVal::ZERO};
    if (fp16_is_inf(a)  || fp16_is_inf(b))
        return {rs ? IntermVal::NEG_INF : IntermVal::POS_INF, rs};

    FP16Norm na = decode_fp16_norm(a);
    FP16Norm nb = decode_fp16_norm(b);
    int32_t  e; uint32_t m;
    mul11(na.sig, na.exp, nb.sig, nb.exp, e, m);

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

        uint32_t mant23 = (m - (1u << (22 - lz))) << (lz + 1);
        return {IntermVal::NORMAL, sc, unb_exp, mant23};
    }
    return {IntermVal::NORMAL, sc, e - 127, m};
}

static IntermVal fp16_mul_fp16_intermed(fp16_t a, fp16_t b) {
    bool rs = fp16_sign(a) ^ fp16_sign(b);

    if (fp16_is_nan(a) || fp16_is_nan(b))                              return {IntermVal::NAN_VAL};
    if ((fp16_is_zero(a) && fp16_is_inf(b)) ||
        (fp16_is_inf(a)  && fp16_is_zero(b)))                          return {IntermVal::NAN_VAL};
    if (fp16_is_zero(a) || fp16_is_zero(b))                            return {IntermVal::ZERO};
    if (fp16_is_inf(a)  || fp16_is_inf(b))
        return {rs ? IntermVal::NEG_INF : IntermVal::POS_INF, rs};

    FP16Norm na = decode_fp16_norm(a);
    FP16Norm nb = decode_fp16_norm(b);
    int32_t  e; uint32_t m;
    mul11(na.sig, na.exp, nb.sig, nb.exp, e, m);

    if (e > 15)  return {rs ? IntermVal::NEG_INF : IntermVal::POS_INF, rs};

    if (e >= -14) return {IntermVal::NORMAL, rs, e, m};

    int shift = -14 - e;
    if (shift >= 24) return {IntermVal::ZERO};

    uint32_t full_sig = (1u << 23) | m;
    uint32_t sub_sig  = full_sig >> shift;
    if (sub_sig == 0) return {IntermVal::ZERO};

    int lead = 31 - __builtin_clz(sub_sig);

    int new_exp = lead - 37;
    uint32_t new_mant;
    if (lead >= 23) new_mant = (sub_sig >> (lead - 23)) & 0x7FFFFFu;
    else            new_mant = (sub_sig << (23 - lead))  & 0x7FFFFFu;
    return {IntermVal::NORMAL, rs, new_exp, new_mant};
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
        uint32_t sig  = m << (lz + 1);
        uint32_t mant23 = (sig & 0x3FFu) << 13;
        return {IntermVal::NORMAL, sc, unb, mant23};
    }

    return {IntermVal::NORMAL, sc, e - 15, m << 13};
}

template<typename T>
static bool check_specials(const IntermVal v[5], T nan_out, T pos_inf_out, T neg_inf_out, T& out) {
    for (int i = 0; i < 5; ++i)
        if (v[i].kind == IntermVal::NAN_VAL) { out = nan_out; return true; }
    bool hp = false, hn = false;
    for (int i = 0; i < 5; ++i) {
        if (v[i].kind == IntermVal::POS_INF) hp = true;
        if (v[i].kind == IntermVal::NEG_INF) hn = true;
    }
    if (hp && hn) { out = nan_out;     return true; }
    if (hp)       { out = pos_inf_out; return true; }
    if (hn)       { out = neg_inf_out; return true; }
    return false;
}

static void align_accum(const IntermVal* v, int n, int sig_bits,
                        bool& sign_out, uint64_t& abs_out, int32_t& max_exp_out) {
    int32_t max_exp = INT32_MIN;
    for (int i = 0; i < n; ++i)
        if (v[i].kind == IntermVal::NORMAL)
            max_exp = std::max(max_exp, v[i].exp);
    max_exp_out = max_exp;

    if (max_exp == INT32_MIN) { sign_out = false; abs_out = 0; return; }

    const int shift_cap = sig_bits + 1;
    int64_t accum = 0;
    for (int i = 0; i < n; ++i) {
        if (v[i].kind != IntermVal::NORMAL) continue;
        int      shift = max_exp - v[i].exp;
        if (shift >= shift_cap) continue;
        uint64_t sig   = ((uint64_t)1 << sig_bits) | v[i].mant;
        uint64_t aligned = sig >> shift;
        if (v[i].sign) accum -= (int64_t)aligned;
        else           accum += (int64_t)aligned;
    }

    sign_out = accum < 0;
    abs_out  = sign_out ? (uint64_t)(-accum) : (uint64_t)accum;
}

static fp32_t accumulate_fp32(const IntermVal v[5]) {
    fp32_t sig_out;
    if (check_specials<fp32_t>(v, NAN_OUT_FP32, 0x7F800000u, 0xFF800000u, sig_out))
        return sig_out;

    bool     rs; uint64_t abs_sum; int32_t max_exp;
    align_accum(v, 5, 23, rs, abs_sum, max_exp);
    if (abs_sum == 0) return 0u;

    int lead = 63 - __builtin_clzll(abs_sum);
    int32_t unb = lead + max_exp - 23;
    int32_t bsd = unb + 127;

    uint32_t mant23;
    if (lead >= 23) mant23 = (uint32_t)((abs_sum >> (lead - 23)) & 0x7FFFFFu);
    else            mant23 = (uint32_t)((abs_sum << (23 - lead))  & 0x7FFFFFu);

    if (bsd >= 255) return rs ? 0xFF800000u : 0x7F800000u;

    if (bsd <= 0) {
        int sub_shift = 1 - bsd;
        if (sub_shift >= 24) return (uint32_t)rs << 31;
        uint32_t sub_mant = ((1u << 23) | mant23) >> sub_shift;
        return ((uint32_t)rs << 31) | sub_mant;
    }

    return ((uint32_t)rs << 31) | ((uint32_t)bsd << 23) | mant23;
}

static fp16_t accumulate_fp16(const IntermVal v[5]) {
    fp16_t sig_out;
    if (check_specials<fp16_t>(v, NAN_OUT_FP16, 0x7C00u, 0xFC00u, sig_out))
        return sig_out;

    bool     rs; uint64_t abs_sum; int32_t max_exp;
    align_accum(v, 5, 23, rs, abs_sum, max_exp);
    if (abs_sum == 0) return 0u;

    int lead = 63 - __builtin_clzll(abs_sum);
    int32_t unb = lead + max_exp - 23;

    uint32_t mant23; bool sticky_norm = false;
    if (lead >= 23) {
        int rs2  = lead - 23;
        mant23   = (uint32_t)((abs_sum >> rs2) & 0x7FFFFFu);
        sticky_norm = (rs2 > 0) && ((abs_sum & ((1ULL << rs2) - 1)) != 0);
    } else {
        mant23 = (uint32_t)((abs_sum << (23 - lead)) & 0x7FFFFFu);
    }
    bool sticky_total = sticky_norm;

    auto rne_round_up = [](uint32_t mant, int round_b, bool sticky) -> bool {
        if (!round_b) return false;
        if (sticky)   return true;
        return (mant & 1u) != 0;
    };

    if (unb >= -14) {
        if (unb > 15) goto overflow_inf;

        uint32_t mant10  = mant23 >> 13;
        int      rnd_bit = (mant23 >> 12) & 1;
        bool     stk     = ((mant23 & 0xFFFu) != 0) || sticky_total;
        if (rne_round_up(mant10, rnd_bit, stk)) {
            mant10++;
            if (mant10 >= (1u << 10)) { mant10 = 0; unb++; }
        }
        if (unb > 15) goto overflow_inf;
        uint32_t e16 = (uint32_t)(unb + 15);
        return (fp16_t)(((uint32_t)rs << 15) | (e16 << 10) | mant10);
    }

    {

        int ts = -1 - unb + 13;

        ts = -(unb + 1);

        uint64_t sig24 = (1u << 23) | mant23;

        if (ts > 23) {

            int rnd_pos = ts - 1;
            int rnd_bit = (rnd_pos < 24) ? (int)((sig24 >> rnd_pos) & 1) : 0;
            bool stk    = sticky_total || (rnd_pos > 0 && (rnd_pos <= 23) &&
                          ((sig24 & ((1ULL << rnd_pos) - 1)) != 0));

            bool ru = rne_round_up(0u, rnd_bit, stk);
            if (!ru) return (fp16_t)((uint32_t)rs << 15);

            return (fp16_t)(((uint32_t)rs << 15) | 1u);
        }

        uint32_t mant10 = (uint32_t)(sig24 >> ts) & 0x3FFu;
        int rnd_bit = (ts > 0) ? (int)((sig24 >> (ts - 1)) & 1) : 0;
        bool stk = sticky_total ||
                   (ts > 1 && ((sig24 & ((1ULL << (ts - 1)) - 1)) != 0));
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

static fp32_t volta_dp4a_fp32_group(const fp16_t a[4], const fp16_t b[4], fp32_t c) {

    for (int i = 0; i < 4; ++i) {
        if (fp16_is_nan(a[i]) || fp16_is_nan(b[i]))           return NAN_OUT_FP32;
        if (fp16_is_zero(a[i]) && fp16_is_inf(b[i]))           return NAN_OUT_FP32;
        if (fp16_is_inf(a[i])  && fp16_is_zero(b[i]))          return NAN_OUT_FP32;
    }
    if (fp32_is_nan(c)) return NAN_OUT_FP32;

    IntermVal v[5];
    for (int i = 0; i < 4; ++i) v[i] = fp16_mul_fp32_intermed(a[i], b[i]);
    v[4] = fp32_to_intermed(c);
    return accumulate_fp32(v);
}

static fp16_t volta_dp4a_fp16_group(const fp16_t a[4], const fp16_t b[4], fp16_t c) {
    for (int i = 0; i < 4; ++i) {
        if (fp16_is_nan(a[i]) || fp16_is_nan(b[i]))            return NAN_OUT_FP16;
        if (fp16_is_zero(a[i]) && fp16_is_inf(b[i]))            return NAN_OUT_FP16;
        if (fp16_is_inf(a[i])  && fp16_is_zero(b[i]))           return NAN_OUT_FP16;
    }
    if (fp16_is_nan(c)) return NAN_OUT_FP16;

    IntermVal v[5];
    for (int i = 0; i < 4; ++i) v[i] = fp16_mul_fp16_intermed(a[i], b[i]);
    v[4] = fp16_to_intermed(c);
    return accumulate_fp16(v);
}

fp32_t volta_dp4a_fp32(const fp16_t* a, const fp16_t* b, size_t len, fp32_t c) {
    static constexpr fp16_t ZERO16 = 0x0000u;
    fp32_t acc = c;
    size_t i = 0;
    for (; i + 4 <= len; i += 4)
        acc = volta_dp4a_fp32_group(a + i, b + i, acc);
    if (i < len) {
        fp16_t pa[4] = {ZERO16, ZERO16, ZERO16, ZERO16};
        fp16_t pb[4] = {ZERO16, ZERO16, ZERO16, ZERO16};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        acc = volta_dp4a_fp32_group(pa, pb, acc);
    }
    return acc;
}

fp16_t volta_dp4a_fp16(const fp16_t* a, const fp16_t* b, size_t len, fp16_t c) {
    static constexpr fp16_t ZERO16 = 0x0000u;
    fp16_t acc = c;
    size_t i = 0;
    for (; i + 4 <= len; i += 4)
        acc = volta_dp4a_fp16_group(a + i, b + i, acc);
    if (i < len) {
        fp16_t pa[4] = {ZERO16, ZERO16, ZERO16, ZERO16};
        fp16_t pb[4] = {ZERO16, ZERO16, ZERO16, ZERO16};
        for (size_t k = i; k < len; ++k) { pa[k-i] = a[k]; pb[k-i] = b[k]; }
        acc = volta_dp4a_fp16_group(pa, pb, acc);
    }
    return acc;
}