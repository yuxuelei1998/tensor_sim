#include "custom_tensor.h"
#include <algorithm>
#include <climits>
#include <vector>

static constexpr uint32_t NAN_OUT_FP32 = 0x7FFFFFFFu;
static constexpr uint16_t NAN_OUT_FP16 = 0x7FFFu;

struct IVal {
    enum Kind : uint8_t { ZERO, NORMAL, POS_INF, NEG_INF, NAN_VAL } kind = ZERO;
    bool     sign = false;
    int32_t  exp  = 0;
    uint64_t mant = 0;
};

struct Elem {
    bool     is_nan  = false;
    bool     is_inf  = false;
    bool     is_zero = false;
    bool     sign    = false;
    int32_t  exp     = 0;
    uint64_t sig     = 0;
    int      sig_bits = 0;
};

static Elem decode_fp16(uint32_t raw) {
    Elem e{};
    e.sign      = fp16_sign((fp16_t)raw);
    e.sig_bits  = 11;
    int      ex = fp16_exp((fp16_t)raw);
    uint32_t m  = fp16_mant((fp16_t)raw);
    if (ex == 31) { if (m) e.is_nan = true; else e.is_inf = true; return e; }
    if (!ex && !m) { e.is_zero = true; return e; }
    if (!ex) {
        int lead = 31 - __builtin_clz(m);
        e.sig = (uint64_t)m << (10 - lead);
        e.exp = lead - 24;
    } else {
        e.sig = (1u << 10) | m;
        e.exp = ex - 15;
    }
    return e;
}

static Elem decode_bf16(uint32_t raw) {
    Elem e{};
    e.sign     = bf16_sign((bf16_t)raw);
    e.sig_bits = 8;
    int      ex = bf16_exp((bf16_t)raw);
    uint32_t m  = bf16_mant((bf16_t)raw);
    if (ex == 255) { if (m) e.is_nan = true; else e.is_inf = true; return e; }
    if (!ex && !m) { e.is_zero = true; return e; }
    if (!ex) {
        int lead = 31 - __builtin_clz(m);
        e.sig = (uint64_t)m << (7 - lead);
        e.exp = lead - 133;
    } else {
        e.sig = (1u << 7) | m;
        e.exp = ex - 127;
    }
    return e;
}

static Elem decode_e5m2(uint32_t raw) {
    Elem e{};
    e.sign     = e5m2_sign((e5m2_t)raw);
    e.sig_bits = 3;
    int      ex = e5m2_exp((e5m2_t)raw);
    uint32_t m  = e5m2_mant((e5m2_t)raw);
    if (ex == 31) { if (m) e.is_nan = true; else e.is_inf = true; return e; }
    if (!ex && !m) { e.is_zero = true; return e; }
    if (!ex) {
        int lead = 31 - __builtin_clz(m);
        e.sig = (uint64_t)m << (2 - lead);
        e.exp = lead - 16;
    } else {
        e.sig = (1u << 2) | m;
        e.exp = ex - 15;
    }
    return e;
}

static Elem decode_e4m3(uint32_t raw) {
    Elem e{};
    e.sign     = e4m3_sign((e4m3_t)raw);
    e.sig_bits = 4;
    int      ex = e4m3_exp((e4m3_t)raw);
    uint32_t m  = e4m3_mant((e4m3_t)raw);
    if ((raw & 0x7Fu) == 0x7Fu) { e.is_nan = true; return e; }
    if (!ex && !m) { e.is_zero = true; return e; }
    if (!ex) {
        int lead = 31 - __builtin_clz(m);
        e.sig = (uint64_t)m << (3 - lead);
        e.exp = lead - 9;
    } else {
        e.sig = (1u << 3) | m;
        e.exp = ex - 7;
    }
    return e;
}

static Elem decode_e2m1(uint32_t raw) {
    Elem e{};
    e.sign     = e2m1_sign((e2m1_t)raw);
    e.sig_bits = 2;
    int      ex = e2m1_exp((e2m1_t)raw);
    uint32_t m  = e2m1_mant((e2m1_t)raw);
    if (!ex && !m) { e.is_zero = true; return e; }
    if (!ex) {
        e.sig = 2u;
        e.exp = -1;
    } else {
        e.sig = (1u << 1) | m;
        e.exp = ex - 1;
    }
    return e;
}

static Elem decode_elem(uint32_t raw, CustomConfig::ABPrec p) {
    switch (p) {
        case CustomConfig::ABPrec::FP16:     return decode_fp16(raw);
        case CustomConfig::ABPrec::BF16:     return decode_bf16(raw);
        case CustomConfig::ABPrec::FP8_E5M2: return decode_e5m2(raw);
        case CustomConfig::ABPrec::FP8_E4M3: return decode_e4m3(raw);
        case CustomConfig::ABPrec::FP4_E2M1: return decode_e2m1(raw);
        default:                             return {};
    }
}

static IVal multiply_w(const Elem& a, const Elem& b, int w) {
    if (a.is_nan || b.is_nan)                            return {IVal::NAN_VAL};
    if ((a.is_zero && b.is_inf) || (a.is_inf && b.is_zero)) return {IVal::NAN_VAL};
    if (a.is_zero || b.is_zero)                          return {IVal::ZERO};
    bool rs = a.sign ^ b.sign;
    if (a.is_inf || b.is_inf)
        return {rs ? IVal::NEG_INF : IVal::POS_INF, rs};

    uint64_t prod = a.sig * b.sig;
    if (!prod) return {IVal::ZERO};

    int lead = 63 - __builtin_clzll(prod);
    int32_t exp_out = a.exp + b.exp
                    + lead - (a.sig_bits - 1) - (b.sig_bits - 1);

    uint64_t frac = prod ^ (1ULL << lead);
    uint64_t mant_w = (lead >= w) ? (frac >> (lead - w))
                                  : (frac << (w  - lead));
    mant_w &= (1ULL << w) - 1;

    return {IVal::NORMAL, rs, exp_out, mant_w};
}

struct SF { bool is_nan, is_zero; uint64_t sig; int32_t exp; };

static SF decode_sf_ue8m0(uint8_t v) {
    if (v == 0xFF) return {true,  false, 0, 0};
    if (v == 0x00) return {false, true,  0, 0};
    return {false, false, 1, (int32_t)v - 127};
}

static SF decode_sf_ue4m3(uint8_t v) {
    if (v == 0x7F || v == 0xFF) return {true, false, 0, 0};
    uint8_t body = v & 0x7Fu;
    if (!body) return {false, true, 0, 0};
    int ef = body >> 3, mf = body & 7;
    if (!ef) {
        if (!mf) return {false, true, 0, 0};
        return {false, false, (uint64_t)mf, -9};
    }
    return {false, false, (uint64_t)(8 | mf), ef - 10};
}

static IVal apply_scale(IVal P, const SF& sa, const SF& sb, int w) {
    if (P.kind == IVal::NAN_VAL)                 return P;
    if (sa.is_nan || sb.is_nan)                  return {IVal::NAN_VAL};
    if (sa.is_zero || sb.is_zero)                return {IVal::ZERO};
    if (P.kind == IVal::ZERO)                    return P;
    if (P.kind == IVal::POS_INF || P.kind == IVal::NEG_INF) return P;


    __uint128_t total = (__uint128_t)((1ULL << w) | P.mant) * sa.sig * sb.sig;
    if (!total) return {IVal::ZERO};

    uint64_t hi = (uint64_t)(total >> 64), lo = (uint64_t)total;
    int lead = hi ? (64 + 63 - __builtin_clzll(hi)) : (63 - __builtin_clzll(lo));

    int32_t new_exp = (int32_t)lead + P.exp - w + sa.exp + sb.exp;

    uint64_t mant_w;
    if (lead >= w) {
        mant_w = (uint64_t)(total >> (lead - w)) & ((1ULL << w) - 1);
    } else {
        uint64_t frac = (uint64_t)(total ^ ((__uint128_t)1 << lead));
        mant_w = (frac << (w - lead)) & ((1ULL << w) - 1);
    }

    return {IVal::NORMAL, P.sign, new_exp, mant_w};
}

static IVal accum_fp32_to_ival(uint32_t raw, int w) {
    if (fp32_is_nan(raw)) return {IVal::NAN_VAL};
    if (fp32_is_inf(raw)) return {fp32_sign(raw) ? IVal::NEG_INF : IVal::POS_INF, fp32_sign(raw)};
    if (fp32_is_zero(raw)) return {IVal::ZERO};

    bool sign = fp32_sign(raw);
    int  ex   = fp32_exp(raw);
    uint32_t m = fp32_mant(raw);

    int32_t  unb_exp;
    uint32_t sig24;
    if (!ex) {
        int lead = 31 - __builtin_clz(m);
        sig24    = m << (23 - lead);
        unb_exp  = lead - 149;
    } else {
        sig24   = (1u << 23) | m;
        unb_exp = ex - 127;
    }

    uint32_t frac23 = sig24 & 0x7FFFFFu;
    uint64_t mant_w = (23 >= w) ? ((uint64_t)frac23 >> (23 - w))
                                 : ((uint64_t)frac23 << (w - 23));
    mant_w &= (1ULL << w) - 1;
    return {IVal::NORMAL, sign, unb_exp, mant_w};
}

static IVal accum_fp16_to_ival(uint16_t raw, int w) {
    if (fp16_is_nan(raw)) return {IVal::NAN_VAL};
    if (fp16_is_inf(raw)) return {fp16_sign(raw) ? IVal::NEG_INF : IVal::POS_INF, fp16_sign(raw)};
    if (fp16_is_zero(raw)) return {IVal::ZERO};

    bool sign = fp16_sign(raw);
    int  ex   = fp16_exp(raw);
    uint32_t m = fp16_mant(raw);

    int32_t  unb_exp;
    uint32_t sig11;
    if (!ex) {
        int lead = 31 - __builtin_clz(m);
        sig11    = m << (10 - lead);
        unb_exp  = lead - 24;
    } else {
        sig11   = (1u << 10) | m;
        unb_exp = ex - 15;
    }

    uint32_t frac10 = sig11 & 0x3FFu;
    uint64_t mant_w = (10 >= w) ? ((uint64_t)frac10 >> (10 - w))
                                 : ((uint64_t)frac10 << (w - 10));
    mant_w &= (1ULL << w) - 1;
    return {IVal::NORMAL, sign, unb_exp, mant_w};
}

static void align_and_sum(const IVal* vals, int n, int w,
                           bool& sign_out, uint64_t& abs_out, int32_t& max_exp_io) {
    int32_t max_exp = INT32_MIN;
    for (int i = 0; i < n; ++i)
        if (vals[i].kind == IVal::NORMAL)
            max_exp = std::max(max_exp, vals[i].exp);

    if (max_exp == INT32_MIN) { sign_out = false; abs_out = 0; return; }
    max_exp_io = max_exp;

    __int128 accum = 0;
    for (int i = 0; i < n; ++i) {
        if (vals[i].kind != IVal::NORMAL) continue;
        int shift = max_exp - vals[i].exp;
        if (shift >= w + 1) continue;
        uint64_t full_sig = (1ULL << w) | vals[i].mant;
        uint64_t aligned  = full_sig >> shift;
        if (vals[i].sign) accum -= (__int128)aligned;
        else              accum += (__int128)aligned;
    }

    sign_out = (accum < 0);
    __uint128_t abs128 = sign_out ? (__uint128_t)(-accum) : (__uint128_t)accum;

    while (abs128 >> 64) { abs128 >>= 1; max_exp_io++; }
    abs_out = (uint64_t)abs128;
}

static void check_specials(const IVal* vals, int n,
                            bool& nan, bool& pos_inf, bool& neg_inf) {
    nan = pos_inf = neg_inf = false;
    for (int i = 0; i < n; ++i) {
        if (vals[i].kind == IVal::NAN_VAL) nan     = true;
        if (vals[i].kind == IVal::POS_INF) pos_inf = true;
        if (vals[i].kind == IVal::NEG_INF) neg_inf = true;
    }
}

static bool rne_round_up(uint64_t mantissa, int rnd_bit, bool sticky) {
    return rnd_bit && (sticky || (mantissa & 1));
}

static uint32_t round_to_fp32(bool sign, uint64_t abs_sig, int lead,
                               int32_t max_exp, int w,
                               CustomConfig::RoundMode rm) {
    if (!abs_sig) return (uint32_t)sign << 31;
    int32_t unb_exp  = max_exp + lead - w;
    int32_t bias_exp = unb_exp + 127;

    uint32_t full24;
    if (lead >= 23) full24 = (uint32_t)(abs_sig >> (lead - 23));
    else            full24 = (uint32_t)(abs_sig << (23 - lead));
    full24 = (full24 & 0xFFFFFFu) | (1u << 23);

    if (bias_exp >= 255) return sign ? 0xFF800000u : 0x7F800000u;

    if (bias_exp > 0) {
        uint32_t mant23 = full24 & 0x7FFFFFu;
        if (rm == CustomConfig::RoundMode::RNE && lead > 23) {
            int drop = lead - 23;
            int rbit = (int)((abs_sig >> (drop - 1)) & 1u);
            bool stk = (drop > 1) && ((abs_sig & ((1ULL << (drop - 1)) - 1)) != 0);
            if (rne_round_up(mant23, rbit, stk)) {
                if (++mant23 >= (1u << 23)) { mant23 = 0; bias_exp++; }
            }
        }
        if (bias_exp >= 255) return sign ? 0xFF800000u : 0x7F800000u;
        return ((uint32_t)sign << 31) | ((uint32_t)bias_exp << 23) | mant23;
    }

    int shift = 1 - bias_exp;
    if (shift > 24) return (uint32_t)sign << 31;
    uint32_t sub = full24 >> shift;
    if (rm == CustomConfig::RoundMode::RNE && shift > 0) {
        int rbit = (int)((full24 >> (shift - 1)) & 1u);
        bool stk = (shift > 1) && ((full24 & ((1u << (shift - 1)) - 1)) != 0);
        if (rne_round_up(sub, rbit, stk)) {
            sub++;
            if (sub >= (1u << 23)) return ((uint32_t)sign << 31) | (1u << 23);
        }
    }
    return ((uint32_t)sign << 31) | sub;
}

static uint16_t round_to_fp16(bool sign, uint64_t abs_sig, int lead,
                               int32_t max_exp, int w,
                               CustomConfig::RoundMode rm) {
    if (!abs_sig) return (uint16_t)((uint32_t)sign << 15);
    int32_t unb_exp  = max_exp + lead - w;
    int32_t bias_exp = unb_exp + 15;

    uint32_t full11;
    if (lead >= 10) full11 = (uint32_t)(abs_sig >> (lead - 10));
    else            full11 = (uint32_t)(abs_sig << (10 - lead));
    full11 = (full11 & 0x7FFu) | (1u << 10);

    if (bias_exp >= 31) return (uint16_t)(((uint32_t)sign << 15) | 0x7C00u);

    if (bias_exp > 0) {
        uint32_t mant10 = full11 & 0x3FFu;
        if (rm == CustomConfig::RoundMode::RNE && lead > 10) {
            int drop = lead - 10;
            int rbit = (int)((abs_sig >> (drop - 1)) & 1u);
            bool stk = (drop > 1) && ((abs_sig & ((1ULL << (drop - 1)) - 1)) != 0);
            if (rne_round_up(mant10, rbit, stk)) {
                if (++mant10 >= (1u << 10)) { mant10 = 0; bias_exp++; }
            }
        }
        if (bias_exp >= 31) return (uint16_t)(((uint32_t)sign << 15) | 0x7C00u);
        return (uint16_t)(((uint32_t)sign << 15) | ((uint32_t)bias_exp << 10) | mant10);
    }

    int shift = 1 - bias_exp;
    if (shift > 11) return (uint16_t)((uint32_t)sign << 15);
    uint32_t sub = full11 >> shift;
    if (rm == CustomConfig::RoundMode::RNE && shift > 0) {
        int rbit = (int)((full11 >> (shift - 1)) & 1u);
        bool stk = (shift > 1) && ((full11 & ((1u << (shift - 1)) - 1)) != 0);
        if (rne_round_up(sub, rbit, stk)) {
            sub++;
            if (sub >= (1u << 10)) return (uint16_t)(((uint32_t)sign << 15) | (1u << 10));
        }
    }
    return (uint16_t)(((uint32_t)sign << 15) | (sub & 0x3FFu));
}

static IVal partial_dot(const uint32_t* a, const uint32_t* b, int count,
                        int w, CustomConfig::ABPrec prec) {
    std::vector<IVal> terms;
    terms.reserve(count);
    for (int i = 0; i < count; ++i)
        terms.push_back(multiply_w(decode_elem(a[i], prec), decode_elem(b[i], prec), w));

    bool nan, pi, ni;
    check_specials(terms.data(), (int)terms.size(), nan, pi, ni);
    if (nan || (pi && ni)) return {IVal::NAN_VAL};
    if (pi)  return {IVal::POS_INF, false};
    if (ni)  return {IVal::NEG_INF, true};

    bool s; uint64_t ab; int32_t me = 0;
    align_and_sum(terms.data(), (int)terms.size(), w, s, ab, me);
    if (!ab) return {IVal::ZERO};

    int lead = 63 - __builtin_clzll(ab);
    int32_t new_exp = me + lead - w;
    uint64_t frac   = ab ^ (1ULL << lead);
    uint64_t mant_w = (lead >= w) ? (frac >> (lead - w)) : (frac << (w - lead));
    mant_w &= (1ULL << w) - 1;
    return {IVal::NORMAL, s, new_exp, mant_w};
}

static uint32_t compute_group(const uint32_t* a, const uint32_t* b,
                               const uint8_t* sa, const uint8_t* sb,
                               int n, uint32_t c, const CustomConfig& cfg) {
    int w = cfg.mant_width;
    IVal c_ival = (cfg.cd_prec == CustomConfig::CDPrec::FP32)
                ? accum_fp32_to_ival(c, w)
                : accum_fp16_to_ival((uint16_t)c, w);

    std::vector<IVal> terms;
    terms.push_back(c_ival);

    if (!cfg.use_scale) {
        for (int i = 0; i < n; ++i)
            terms.push_back(multiply_w(decode_elem(a[i], cfg.ab_prec),
                                       decode_elem(b[i], cfg.ab_prec), w));
    } else {
        int sg       = cfg.scale_group;
        int n_groups = (n + sg - 1) / sg;
        for (int g = 0; g < n_groups; ++g) {
            int start = g * sg;
            int end   = std::min(start + sg, n);
            IVal P    = partial_dot(a + start, b + start, end - start, w, cfg.ab_prec);

            SF sfa = (cfg.scale_type == CustomConfig::ScaleType::UE8M0)
                   ? decode_sf_ue8m0(sa[g]) : decode_sf_ue4m3(sa[g]);
            SF sfb = (cfg.scale_type == CustomConfig::ScaleType::UE8M0)
                   ? decode_sf_ue8m0(sb[g]) : decode_sf_ue4m3(sb[g]);

            terms.push_back(apply_scale(P, sfa, sfb, w));
        }
    }

    bool nan, pi, ni;
    check_specials(terms.data(), (int)terms.size(), nan, pi, ni);

    if (cfg.cd_prec == CustomConfig::CDPrec::FP32) {
        if (nan || (pi && ni)) return NAN_OUT_FP32;
        if (pi) return 0x7F800000u;
        if (ni) return 0xFF800000u;
        bool s; uint64_t ab; int32_t me = 0;
        align_and_sum(terms.data(), (int)terms.size(), w, s, ab, me);
        if (!ab) return 0u;
        int lead = 63 - __builtin_clzll(ab);
        return round_to_fp32(s, ab, lead, me, w, cfg.round_mode);
    } else {
        if (nan || (pi && ni)) return NAN_OUT_FP16;
        if (pi) return 0x7C00u;
        if (ni) return 0xFC00u;
        bool s; uint64_t ab; int32_t me = 0;
        align_and_sum(terms.data(), (int)terms.size(), w, s, ab, me);
        if (!ab) return 0u;
        int lead = 63 - __builtin_clzll(ab);
        return round_to_fp16(s, ab, lead, me, w, cfg.round_mode);
    }
}

uint32_t custom_dot_product(const uint32_t* a, const uint32_t* b,
                             const uint8_t* sa, const uint8_t* sb,
                             size_t len, uint32_t c,
                             const CustomConfig& cfg) {
    int n     = cfg.dp_width;
    int w     = cfg.mant_width;
    (void)w;
    int sg    = cfg.use_scale ? cfg.scale_group : 1;
    int n_sf  = cfg.use_scale ? ((n + sg - 1) / sg) : 0;

    uint32_t acc = c;
    std::vector<uint32_t> pa(n, 0), pb(n, 0);
    std::vector<uint8_t>  psa(n_sf, 0), psb(n_sf, 0);

    size_t i = 0, g = 0;
    while (i < len) {
        size_t remaining = std::min((size_t)n, len - i);
        for (size_t k = 0; k < remaining; ++k) { pa[k] = a[i+k]; pb[k] = b[i+k]; }
        for (int  k = (int)remaining; k < n; ++k) { pa[k] = 0; pb[k] = 0; }

        if (cfg.use_scale) {
            int sf_avail = (int)std::min((size_t)n_sf,
                           (len - i + sg - 1) / sg);
            for (int k = 0; k < sf_avail; ++k) { psa[k] = sa[g+k]; psb[k] = sb[g+k]; }
            for (int k = sf_avail; k < n_sf; ++k) { psa[k] = 0; psb[k] = 0; }
        }

        acc = compute_group(pa.data(), pb.data(),
                            cfg.use_scale ? psa.data() : nullptr,
                            cfg.use_scale ? psb.data() : nullptr,
                            n, acc, cfg);

        i += remaining;
        g += n_sf;
    }
    return acc;
}