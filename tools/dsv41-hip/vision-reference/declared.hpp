#pragma once
// From hyukudan/strix-halo-engine-lab's include/strix/ds41_moe.hpp, Apache-2.0 (THIRD_PARTY_NOTICES.md).
#include <cstdint>

// Only its declared_exp, declared_sqrt and STRIX_DS41_HD, code unchanged (comments shortened).

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_DS41_HD __host__ __device__
#else
#define STRIX_DS41_HD
#endif

namespace strix::ds41 {

namespace detail {
STRIX_DS41_HD inline float pow2(int k) {
    return __builtin_bit_cast(float, static_cast<std::uint32_t>(k + 127) << 23U);
}
} // namespace detail

// exp(x) from basic IEEE operations only: Cody-Waite by ln 2, degree-7 Taylor (Horner), two-step 2^k scaling.
STRIX_DS41_HD inline float declared_exp(float x) {
    if (!(x == x)) return x;
    if (x > 88.0F) return __builtin_huge_valf();
    if (x < -103.0F) return 0.0F;
    const float n = __builtin_rintf(x * 1.44269504F);
    const float r = (x - n * 0.693359375F) - n * -2.12194440e-4F;
    float p = 1.98412698e-4F;
    p = p * r + 1.38888889e-3F;
    p = p * r + 8.33333333e-3F;
    p = p * r + 4.16666667e-2F;
    p = p * r + 1.66666667e-1F;
    p = p * r + 0.5F;
    p = p * r + 1.0F;
    p = p * r + 1.0F;
    const int k = static_cast<int>(n), k1 = k / 2, k2 = k - k1;
    return (p * detail::pow2(k1)) * detail::pow2(k2);
}

// Correctly rounded square root (on the device through double precision: its float rounding is the correct root).
STRIX_DS41_HD inline float declared_sqrt(float x) {
#if defined(__HIP_DEVICE_COMPILE__)
    return static_cast<float>(__dsqrt_rn(static_cast<double>(x)));
#elif defined(__CUDA_ARCH__)
    return __fsqrt_rn(x);  // IEEE round-to-nearest single-precision root on NVIDIA devices
#else
    return __builtin_sqrtf(x);
#endif
}

} // namespace strix::ds41
