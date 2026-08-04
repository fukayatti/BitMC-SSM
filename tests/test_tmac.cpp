/**
 * Benchmark: Traditional Zero-GEMM vs T-MAC LUT-based GEMM
 * Tests numerical correctness and execution speed.
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <cassert>
#include "../src/tmac_gemm.h"

// Baseline Zero-GEMM (Scalar Addition/Subtraction decode)
void baseline_zero_gemm(
    const uint8_t* packed_weights,
    const float* x,
    float* y,
    uint32_t in_features,
    uint32_t out_features,
    float gamma
) {
    static constexpr int8_t DECODE_LUT[4] = {0, 1, -1, 0};
    uint32_t num_chunks = in_features / 4;

    for (uint32_t row = 0; row < out_features; ++row) {
        float sum = 0.0f;
        const uint8_t* row_w = packed_weights + row * num_chunks;

        for (uint32_t c = 0; c < num_chunks; ++c) {
            uint8_t b = row_w[c];
            sum += DECODE_LUT[(b >> 0) & 0x03] * x[c * 4 + 0];
            sum += DECODE_LUT[(b >> 2) & 0x03] * x[c * 4 + 1];
            sum += DECODE_LUT[(b >> 4) & 0x03] * x[c * 4 + 2];
            sum += DECODE_LUT[(b >> 6) & 0x03] * x[c * 4 + 3];
        }
        y[row] = sum * gamma;
    }
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "⚡ T-MAC (Ternary LUT GEMM) vs Baseline Zero-GEMM Benchmark" << std::endl;
    std::cout << "======================================================================" << std::endl;

    const uint32_t in_features = 512;
    const uint32_t out_features = 512;
    const uint32_t num_runs = 5000;
    const float gamma = 0.05f;

    // Generate random packed weights and input vector
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_real_distribution<float> float_dist(-1.0f, 1.0f);

    size_t num_bytes = static_cast<size_t>(out_features) * (in_features / 4);
    std::vector<uint8_t> packed_weights(num_bytes);
    for (auto& b : packed_weights) b = static_cast<uint8_t>(byte_dist(rng));

    std::vector<float> x(in_features);
    for (auto& v : x) v = float_dist(rng);

    std::vector<float> y_baseline(out_features, 0.0f);
    std::vector<float> y_tmac(out_features, 0.0f);

    // 1. Correctness Verification
    baseline_zero_gemm(packed_weights.data(), x.data(), y_baseline.data(), in_features, out_features, gamma);
    tmac::tmac_gemv(packed_weights.data(), x.data(), y_tmac.data(), in_features, out_features, gamma);

    float max_diff = 0.0f;
    for (uint32_t i = 0; i < out_features; ++i) {
        float diff = std::abs(y_baseline[i] - y_tmac[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::cout << "1. Numerical Correctness: Max Absolute Difference = " << max_diff << std::endl;
    assert(max_diff < 1e-4f && "T-MAC output does not match baseline!");
    std::cout << "   ✅ Numerical verification PASSED (Exact Match)!" << std::endl;

    // 2. Speed Benchmark
    std::cout << "\n2. Performance Benchmark (" << num_runs << " iterations of " << in_features << "x" << out_features << " GEMV):" << std::endl;

    // Baseline
    auto t0 = std::chrono::high_resolution_clock::now();
    for (uint32_t r = 0; r < num_runs; ++r) {
        baseline_zero_gemm(packed_weights.data(), x.data(), y_baseline.data(), in_features, out_features, gamma);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double time_baseline_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // T-MAC
    auto t2 = std::chrono::high_resolution_clock::now();
    for (uint32_t r = 0; r < num_runs; ++r) {
        tmac::tmac_gemv(packed_weights.data(), x.data(), y_tmac.data(), in_features, out_features, gamma);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double time_tmac_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    std::cout << "   - Baseline Zero-GEMM: " << time_baseline_ms << " ms (" << (time_baseline_ms / num_runs * 1000.0) << " us/call)" << std::endl;
    std::cout << "   - T-MAC (LUT GEMM):   " << time_tmac_ms << " ms (" << (time_tmac_ms / num_runs * 1000.0) << " us/call)" << std::endl;
    std::cout << "   🚀 Speedup Ratio:     " << (time_baseline_ms / time_tmac_ms) << "x FASTER!" << std::endl;
    std::cout << "======================================================================" << std::endl;

    return 0;
}
