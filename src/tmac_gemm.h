/**
 * T-MAC (Ternary Multiply-Accumulate / LUT-based GEMM Kernel)
 * Reference: Microsoft Research (2024) / High-Performance 1.58-bit CPU Kernel
 *
 * Replaces arithmetic addition/subtraction loops with L1-Cache Look-Up Tables (LUT).
 * Precomputes activations into 4-weight combinations (16 to 256 patterns per chunk),
 * reducing matrix-vector multiplication to pure memory table lookups.
 */

#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace tmac {

// Weight decode table for 2-bit values: 00 -> 0, 01 -> +1, 10 -> -1, 11 -> 0
static constexpr float DECODE_FLOAT[4] = {0.0f, 1.0f, -1.0f, 0.0f};

/**
 * Precomputes LUT for a 4-element input chunk (x0, x1, x2, x3).
 * Generates all 256 possible combinations of (w0*x0 + w1*x1 + w2*x2 + w3*x3).
 * Size: 256 * sizeof(float) = 1024 bytes per 4-tuple.
 */
inline void precompute_lut_4(const float* x, float* lut_table_256) {
    float w0_val[4] = {0.0f, x[0], -x[0], 0.0f};
    float w1_val[4] = {0.0f, x[1], -x[1], 0.0f};
    float w2_val[4] = {0.0f, x[2], -x[2], 0.0f};
    float w3_val[4] = {0.0f, x[3], -x[3], 0.0f};

    for (int b = 0; b < 256; ++b) {
        int b0 = (b >> 0) & 0x03;
        int b1 = (b >> 2) & 0x03;
        int b2 = (b >> 4) & 0x03;
        int b3 = (b >> 6) & 0x03;
        lut_table_256[b] = w0_val[b0] + w1_val[b1] + w2_val[b2] + w3_val[b3];
    }
}

/**
 * High-Speed T-MAC Matrix-Vector Multiplication: y = gamma * (W_packed * x)
 * 
 * @param packed_weights 2-bit packed weight matrix [out_features, in_features / 4]
 * @param x Input activation vector [in_features]
 * @param y Output vector [out_features]
 * @param in_features Dimension of input (must be multiple of 4)
 * @param out_features Dimension of output
 * @param gamma Scaling factor
 */
inline void tmac_gemv(
    const uint8_t* packed_weights,
    const float* x,
    float* y,
    uint32_t in_features,
    uint32_t out_features,
    float gamma
) {
    const uint32_t num_chunks = in_features / 4;
    
    // Allocate L1-resident LUT buffer: num_chunks * 256 floats
    // For in_features=512: 128 chunks * 256 floats = 128 KB (fits comfortably in L2/L1 cache)
    std::vector<float> lut_buffer(num_chunks * 256);

    // Step 1: Precompute LUT for all 4-element chunks of input x
    for (uint32_t c = 0; c < num_chunks; ++c) {
        precompute_lut_4(x + c * 4, lut_buffer.data() + c * 256);
    }

    // Step 2: Ultra-Fast Matrix-Vector Product via Table Lookups
    #pragma omp parallel for if(out_features >= 16384) schedule(static)
    for (uint32_t row = 0; row < out_features; ++row) {
        const uint8_t* row_weights = packed_weights + (row * num_chunks);
        float sum = 0.0f;

        // Unroll 4x for high instruction-level parallelism (ILP)
        uint32_t c = 0;
        for (; c + 3 < num_chunks; c += 4) {
            sum += lut_buffer[(c + 0) * 256 + row_weights[c + 0]];
            sum += lut_buffer[(c + 1) * 256 + row_weights[c + 1]];
            sum += lut_buffer[(c + 2) * 256 + row_weights[c + 2]];
            sum += lut_buffer[(c + 3) * 256 + row_weights[c + 3]];
        }
        for (; c < num_chunks; ++c) {
            sum += lut_buffer[c * 256 + row_weights[c]];
        }

        y[row] = sum * gamma;
    }
}

} // namespace tmac
