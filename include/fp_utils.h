#pragma once
#include <cstdint>
#include <cstring>

using fp16_t = uint16_t;
using fp32_t = uint32_t;

inline bool     fp16_sign  (fp16_t x) { return (x >> 15) & 1u; }
inline int      fp16_exp   (fp16_t x) { return (x >> 10) & 0x1F; }
inline uint32_t fp16_mant  (fp16_t x) { return x & 0x3FFu; }

inline bool fp16_is_nan    (fp16_t x) { return fp16_exp(x) == 31 && fp16_mant(x) != 0; }
inline bool fp16_is_inf    (fp16_t x) { return fp16_exp(x) == 31 && fp16_mant(x) == 0; }
inline bool fp16_is_zero   (fp16_t x) { return (x & 0x7FFFu) == 0; }
inline bool fp16_is_subnorm(fp16_t x) { return fp16_exp(x) == 0 && fp16_mant(x) != 0; }

inline bool     fp32_sign  (fp32_t x) { return (x >> 31) & 1u; }
inline int      fp32_exp   (fp32_t x) { return (x >> 23) & 0xFF; }
inline uint32_t fp32_mant  (fp32_t x) { return x & 0x7FFFFFu; }

inline bool fp32_is_nan    (fp32_t x) { return fp32_exp(x) == 0xFF && fp32_mant(x) != 0; }
inline bool fp32_is_inf    (fp32_t x) { return fp32_exp(x) == 0xFF && fp32_mant(x) == 0; }
inline bool fp32_is_zero   (fp32_t x) { return (x & 0x7FFFFFFFu) == 0; }

inline float bits_to_float(fp32_t b) {
    float f; std::memcpy(&f, &b, 4); return f;
}
inline fp32_t float_to_bits(float f) {
    fp32_t b; std::memcpy(&b, &f, 4); return b;
}

inline float fp16_to_float(fp16_t h) {
    uint32_t s = (h >> 15) & 1u;
    int      e = (h >> 10) & 0x1F;
    uint32_t m = h & 0x3FFu;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) {
            bits = s << 31;
        } else {
            int lz = __builtin_clz(m) - 22;
            int e32 = 127 + (-15 - lz);
            uint32_t sig = m << (lz + 1);
            uint32_t mant32 = (sig & 0x3FFu) << 13;
            bits = (s << 31) | ((uint32_t)e32 << 23) | mant32;
        }
    } else if (e == 31) {
        bits = (s << 31) | 0x7F800000u | (m << 13);
    } else {
        uint32_t mant32 = m << 13;
        bits = (s << 31) | ((uint32_t)(127 + e - 15) << 23) | mant32;
    }
    return bits_to_float(bits);
}

using bf16_t = uint16_t;

inline bool     bf16_sign  (bf16_t x) { return (x >> 15) & 1u; }
inline int      bf16_exp   (bf16_t x) { return (x >> 7) & 0xFF; }
inline uint32_t bf16_mant  (bf16_t x) { return x & 0x7Fu; }

inline bool bf16_is_nan    (bf16_t x) { return bf16_exp(x) == 0xFF && bf16_mant(x) != 0; }
inline bool bf16_is_inf    (bf16_t x) { return bf16_exp(x) == 0xFF && bf16_mant(x) == 0; }
inline bool bf16_is_zero   (bf16_t x) { return (x & 0x7FFFu) == 0; }
inline bool bf16_is_subnorm(bf16_t x) { return bf16_exp(x) == 0 && bf16_mant(x) != 0; }

inline float bf16_to_float(bf16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    return bits_to_float(bits);
}

inline bf16_t float_to_bf16(float f) {
    fp32_t bits = float_to_bits(f);
    uint32_t e = (bits >> 23) & 0xFF;
    uint32_t m = bits & 0x7FFFFFu;

    if (e == 0xFF && m != 0) return (bf16_t)((bits >> 16) | 0x0040u);
    return (bf16_t)(bits >> 16);
}

using e5m2_t = uint8_t;

inline bool     e5m2_sign  (e5m2_t x) { return (x >> 7) & 1u; }
inline int      e5m2_exp   (e5m2_t x) { return (x >> 2) & 0x1F; }
inline uint32_t e5m2_mant  (e5m2_t x) { return x & 0x3u; }

inline bool e5m2_is_nan    (e5m2_t x) { return e5m2_exp(x) == 31 && e5m2_mant(x) != 0; }
inline bool e5m2_is_inf    (e5m2_t x) { return e5m2_exp(x) == 31 && e5m2_mant(x) == 0; }
inline bool e5m2_is_zero   (e5m2_t x) { return (x & 0x7Fu) == 0; }
inline bool e5m2_is_subnorm(e5m2_t x) { return e5m2_exp(x) == 0 && e5m2_mant(x) != 0; }

inline float e5m2_to_float(e5m2_t x) {
    uint32_t s = (x >> 7) & 1u;
    int      e = (x >> 2) & 0x1F;
    uint32_t m = x & 0x3u;
    if (e == 31) {

        return bits_to_float((s << 31) | 0x7F800000u | (m << 21));
    }
    if (e == 0) {
        if (m == 0) return bits_to_float(s << 31);

        int lead = (m & 2) ? 1 : 0;
        uint32_t sig = m << (2 - lead);
        int e32 = 127 + (-15 - (1 - lead));
        uint32_t mant32 = (sig & 0x3u) << 21;
        return bits_to_float((s << 31) | ((uint32_t)e32 << 23) | mant32);
    }

    uint32_t mant32 = m << 21;
    return bits_to_float((s << 31) | ((uint32_t)(127 + e - 15) << 23) | mant32);
}

using e4m3_t = uint8_t;

inline bool     e4m3_sign  (e4m3_t x) { return (x >> 7) & 1u; }
inline int      e4m3_exp   (e4m3_t x) { return (x >> 3) & 0xF; }
inline uint32_t e4m3_mant  (e4m3_t x) { return x & 0x7u; }

inline bool e4m3_is_nan    (e4m3_t x) { return (x & 0x7Fu) == 0x7Fu; }
inline bool e4m3_is_inf    (e4m3_t)   { return false; }
inline bool e4m3_is_zero   (e4m3_t x) { return (x & 0x7Fu) == 0; }
inline bool e4m3_is_subnorm(e4m3_t x) { return e4m3_exp(x) == 0 && e4m3_mant(x) != 0; }

inline float e4m3_to_float(e4m3_t x) {
    uint32_t s = (x >> 7) & 1u;
    int      e = (x >> 3) & 0xF;
    uint32_t m = x & 0x7u;
    if (e == 15 && m == 7) return bits_to_float((s << 31) | 0x7FC00000u);
    if (e == 0) {
        if (m == 0) return bits_to_float(s << 31);

        float val = (float)m * (1.0f / 64.0f);

        if (s) val = -val;
        return val;
    }

    uint32_t mant32 = m << 20;
    return bits_to_float((s << 31) | ((uint32_t)(127 + e - 7) << 23) | mant32);
}

using e2m1_t = uint8_t;

inline bool     e2m1_sign  (e2m1_t x) { return (x >> 3) & 1u; }
inline int      e2m1_exp   (e2m1_t x) { return (x >> 1) & 0x3; }
inline uint32_t e2m1_mant  (e2m1_t x) { return x & 0x1u; }

inline bool e2m1_is_nan    (e2m1_t)   { return false; }
inline bool e2m1_is_inf    (e2m1_t)   { return false; }
inline bool e2m1_is_zero   (e2m1_t x) { return (x & 0x7u) == 0; }
inline bool e2m1_is_subnorm(e2m1_t x) { return e2m1_exp(x) == 0 && e2m1_mant(x) != 0; }

inline float e2m1_to_float(e2m1_t x) {
    uint32_t s = (x >> 3) & 1u;
    int      e = (x >> 1) & 0x3;
    uint32_t m = x & 0x1u;
    if (e == 0) {
        if (m == 0) return bits_to_float(s << 31);
        float val = 0.5f;
        if (s) val = -val;
        return val;
    }
    uint32_t mant32 = m << 22;
    return bits_to_float((s << 31) | ((uint32_t)(127 + e - 1) << 23) | mant32);
}

inline fp16_t float_to_fp16(float f) {
    fp32_t fb = float_to_bits(f);
    uint32_t s = fb >> 31;
    int      e = (fb >> 23) & 0xFF;
    uint32_t m = fb & 0x7FFFFFu;
    if (e == 0xFF) {
        return (fp16_t)((s << 15) | 0x7C00u | (m ? 0x200u : 0u));
    }
    int e16 = e - 127 + 15;
    if (e16 >= 31)  return (fp16_t)((s << 15) | 0x7C00u);
    if (e16 <= 0) {
        if (e16 < -10) return (fp16_t)(s << 15);
        uint32_t sig = (1u << 10) | (m >> 13);
        return (fp16_t)((s << 15) | (sig >> (1 - e16)));
    }
    return (fp16_t)((s << 15) | ((uint32_t)e16 << 10) | (m >> 13));
}