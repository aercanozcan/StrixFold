#pragma once
// The vision tower's element-wise declared arithmetic: basic IEEE operations only, so host and device agree bitwise.
#include "declared.hpp"  // declared_exp, declared_sqrt, STRIX_DS41_HD

#include <cstdint>

namespace strix::deepseek41::vision {

// Attention keys are taken in blocks of this many (online softmax).
inline constexpr unsigned kVisionKeyBlock = 64;

// Round to BF16, nearest even (torch's .to(torch.bfloat16)); NaN stays a quiet NaN.
STRIX_DS41_HD inline std::uint16_t bf16_round(float v) {
    std::uint32_t u = __builtin_bit_cast(std::uint32_t, v);
    if ((u & 0x7FFFFFFFU) > 0x7F800000U) return static_cast<std::uint16_t>((u >> 16U) | 0x40U);
    u += 0x7FFFU + ((u >> 16U) & 1U);
    return static_cast<std::uint16_t>(u >> 16U);
}
STRIX_DS41_HD inline float bf16_float(std::uint16_t b) { return __builtin_bit_cast(float, static_cast<std::uint32_t>(b) << 16U); }

// erff of fdlibm as musl carries it (FreeBSD s_erff.c). Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved. Developed at SunPro, a Sun Microsystems, Inc. business. Permission to use, copy, modify, and distribute this software is freely granted, provided that this notice is preserved.
STRIX_DS41_HD inline float declared_erf(float x) {
    const std::uint32_t bits = __builtin_bit_cast(std::uint32_t, x);
    const bool negative = (bits >> 31U) != 0U;
    const std::uint32_t ix = bits & 0x7FFFFFFFU;
    if (ix > 0x7F800000U) return x;                           // NaN
    if (ix == 0x7F800000U) return negative ? -1.0F : 1.0F;    // +-inf
    if (ix < 0x3F580000U) {                                   // |x| < 0.84375
        if (ix < 0x31800000U) return 0.125F * (8.0F * x + 1.0270333290e+00F * x);
        const float z = x * x;
        const float r = 1.2837916613e-01F + z * (-3.2504209876e-01F + z * (-2.8481749818e-02F + z * (-5.7702702470e-03F + z * -2.3763017452e-05F)));
        const float s = 1.0F + z * (3.9791721106e-01F + z * (6.5022252500e-02F + z * (5.0813062117e-03F + z * (1.3249473704e-04F + z * -3.9602282413e-06F))));
        const float y = r / s;
        return x + x * y;
    }
    float y;
    const float ax = negative ? -x : x;
    if (ix < 0x3FA00000U) {  // |x| < 1.25
        const float s = ax - 1.0F;
        const float P = -2.3621185683e-03F + s * (4.1485610604e-01F + s * (-3.7220788002e-01F + s * (3.1834661961e-01F + s * (-1.1089469492e-01F + s * (3.5478305072e-02F + s * -2.1663755178e-03F)))));
        const float Q = 1.0F + s * (1.0642088205e-01F + s * (5.4039794207e-01F + s * (7.1828655899e-02F + s * (1.2617121637e-01F + s * (1.3637083583e-02F + s * 1.1984500103e-02F)))));
        y = 8.4506291151e-01F + P / Q;  // 1 - erfc1 = erx + P/Q
    } else if (ix < 0x40C00000U) {  // |x| < 6
        const float s = 1.0F / (ax * ax);
        float R, S;
        if (ix < 0x4036DB6DU) {  // |x| < 1/0.35
            R = -9.8649440333e-03F + s * (-6.9385856390e-01F + s * (-1.0558626175e+01F + s * (-6.2375331879e+01F + s * (-1.6239666748e+02F + s * (-1.8460508728e+02F + s * (-8.1287437439e+01F + s * -9.8143291473e+00F))))));
            S = 1.0F + s * (1.9651271820e+01F + s * (1.3765776062e+02F + s * (4.3456588745e+02F + s * (6.4538726807e+02F + s * (4.2900814819e+02F + s * (1.0863500214e+02F + s * (6.5702495575e+00F + s * -6.0424413532e-02F)))))));
        } else {
            R = -9.8649431020e-03F + s * (-7.9928326607e-01F + s * (-1.7757955551e+01F + s * (-1.6063638306e+02F + s * (-6.3756646729e+02F + s * (-1.0250950928e+03F + s * -4.8351919556e+02F)))));
            S = 1.0F + s * (3.0338060379e+01F + s * (3.2579251099e+02F + s * (1.5367296143e+03F + s * (3.1998581543e+03F + s * (2.5530502930e+03F + s * (4.7452853394e+02F + s * -2.2440952301e+01F))))));
        }
        const float z = __builtin_bit_cast(float, __builtin_bit_cast(std::uint32_t, ax) & 0xFFFFE000U);
        const float erfc = ds41::declared_exp(-z * z - 0.5625F) * ds41::declared_exp((z - ax) * (z + ax) + R / S) / ax;
        y = 1.0F - erfc;
    } else {
        y = 1.0F;
    }
    return negative ? -y : y;
}

// Wave-order sum: lane l of 32 sums elements l, l + 32, ... ascending; the butterfly v + v[l ^ d], d = 16 .. 1.
inline float wave_sum_host(const float* partial) {
    float p[32], q[32];
    for (int l = 0; l < 32; ++l) p[l] = partial[l];
    for (int d = 16; d != 0; d /= 2) {
        for (int l = 0; l < 32; ++l) q[l] = p[l] + p[l ^ d];
        for (int l = 0; l < 32; ++l) p[l] = q[l];
    }
    return p[0];
}

// GELU (exact form, torch's CUDA order): (x * 0.5) * (1 + erf(x * sqrt(1/2))).
STRIX_DS41_HD inline float vision_gelu(float x) { return (x * 0.5F) * (1.0F + declared_erf(x * 0.70710678118654752F)); }
// SiLU: g / (1 + exp(-g)).
STRIX_DS41_HD inline float vision_silu(float g) { return g / (1.0F + ds41::declared_exp(-g)); }
// The MLP's product: bf16(bf16(silu(gate)) * up), gate and up BF16.
STRIX_DS41_HD inline std::uint16_t vision_silu_mul(std::uint16_t gate, std::uint16_t up) {
    return bf16_round(bf16_float(bf16_round(vision_silu(bf16_float(gate)))) * bf16_float(up));
}
// RMSNorm of one element given the row's inverse root r: bf16(w * (x * r)).
STRIX_DS41_HD inline std::uint16_t vision_norm(std::uint16_t x, float r, std::uint16_t w) { return bf16_round(bf16_float(w) * (bf16_float(x) * r)); }
// The row's inverse root from its sum of squares: 1 / sqrt(ss / n + eps).
STRIX_DS41_HD inline float vision_inv_root(float ss, float n, float eps) { return 1.0F / ds41::declared_sqrt(ss / n + eps); }
// Rotary pair (rotate-half): o1 = x1 c - x2 s, o2 = x2 c + x1 s, each product and sum one IEEE operation.
STRIX_DS41_HD inline void vision_rotate(float x1, float x2, float c, float s, float& o1, float& o2) {
    const float a = x1 * c, b = x2 * s, d = x2 * c, e = x1 * s;
    o1 = a - b;
    o2 = d + e;
}

}  // namespace strix::deepseek41::vision
