#pragma once

#include <vector>
#include <cmath>
#include <immintrin.h>
#include <cstdint>
#include <algorithm>
#include <cstring>

namespace hadamard {

/**
 * Fast Walsh-Hadamard Transform (FWHT) in-place.
 * Time Complexity: O(N log N) using ONLY additions and subtractions.
 * Space Complexity: O(1) in-place.
 */
inline void fwht(float* data, size_t n) {
    // 1. Butterfly additions & subtractions
    for (size_t h = 1; h < n; h *= 2) {
        for (size_t i = 0; i < n; i += 2 * h) {
            for (size_t j = i; j < i + h; ++j) {
                float x = data[j];
                float y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
    }

    // 2. SIMD Vectorized Scaling by 1 / sqrt(n)
    float scale = 1.0f / std::sqrt(static_cast<float>(n));
    size_t i = 0;
    __m256 vscale = _mm256_set1_ps(scale);
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        v = _mm256_mul_ps(v, vscale);
        _mm256_storeu_ps(data + i, v);
    }
    for (; i < n; ++i) {
        data[i] *= scale;
    }
}

/**
 * Native INT4 Activation Quantization (BitNet v2)
 * Converts outlier-free Hadamard activations into 4-bit integer codes.
 * Returns scale factor gamma.
 */
inline float quantize_act_int4(const float* x, int8_t* out_int4, size_t n) {
    float sum_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum_abs += std::abs(x[i]);
    }
    float gamma = std::max(sum_abs / n, 1e-5f);
    float scale = 7.0f / (gamma * 1.5f);
    float inv_scale = (gamma * 1.5f) / 7.0f;

    for (size_t i = 0; i < n; ++i) {
        int v = static_cast<int>(std::round(x[i] * scale));
        out_int4[i] = static_cast<int8_t>(std::clamp(v, -8, 7));
    }
    return inv_scale;
}

} // namespace hadamard
