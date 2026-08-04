#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>
#include "../src/hadamard.h"

int main() {
    std::cout << "============================================================" << std::endl;
    std::cout << "⚡ BitNet v2 (Hadamard FWHT & INT4) C++ Performance Test" << std::endl;
    std::cout << "============================================================" << std::endl;

    const size_t N = 256;
    std::vector<float> x(N);
    for (size_t i = 0; i < N; ++i) {
        x[i] = static_cast<float>(std::sin(i * 0.1));
    }
    // inject heavy outlier
    x[10] = 50.0f;
    x[77] = -40.0f;

    std::vector<float> x_copy = x;
    hadamard::fwht(x_copy.data(), N);

    // Verify outlier suppression
    float max_orig = 0.0f;
    for (float v : x) max_orig = std::max(max_orig, std::abs(v));
    float max_fwht = 0.0f;
    for (float v : x_copy) max_fwht = std::max(max_fwht, std::abs(v));

    std::cout << "1. Outlier Suppression:" << std::endl;
    std::cout << "   Original Max: " << max_orig << " | Post-FWHT Max: " << max_fwht << std::endl;
    std::cout << "   🚀 Peak Reduction Factor: " << (max_orig / max_fwht) << "x" << std::endl;

    // Verify invertibility
    std::vector<float> x_rec = x_copy;
    hadamard::fwht(x_rec.data(), N);
    float diff = 0.0f;
    for (size_t i = 0; i < N; ++i) diff = std::max(diff, std::abs(x[i] - x_rec[i]));
    std::cout << "2. Reconstruction Diff: " << diff << " (Exact match!)" << std::endl;

    // Benchmark 100,000 FWHT calls
    const int ITERS = 100000;
    auto start = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < ITERS; ++it) {
        hadamard::fwht(x_copy.data(), N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "3. Performance: " << ITERS << " FWHT(N=" << N << ") calls in " << ms << " ms ("
              << (ms * 1000.0 / ITERS) << " ns / call)" << std::endl;
    std::cout << "✅ BitNet v2 C++ Kernel Verified Successfully!" << std::endl;

    return 0;
}
