/**
 * Hierarchical Cache & Dynamic Memory Placer (L1 / L2 / L3 / DRAM)
 * 
 * Hardware-Aware 3-Tier Cache Architecture:
 *  - Tier 1 (L1 Cache / ~64KB): Recent window tokens (L <= 32) in raw FP32 for zero-latency instant access.
 *  - Tier 2 (L2/L3 Cache / ~16-32MB): Mid-range segment states (32 < L <= 2048) in TurboQuant 3-bit.
 *  - Tier 3 (DRAM / Main Memory): Long-range archive (L > 2048) with 1-bit QJL bitwise POPCNT search.
 */

#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace cache {

// Tier 2: TurboQuant 3-bit Compressed State
struct TurboQuant3State {
    float norm;
    std::vector<uint8_t> packed_codes; // 2 codes (3-bit each) per byte or packed array
    
    // Dequantize back to float vector of dimension d
    void dequantize(uint32_t dim, float* out, const float* centroids_8) const {
        for (uint32_t i = 0; i < dim; ++i) {
            uint8_t code = (i % 2 == 0) ? (packed_codes[i / 2] & 0x07) : ((packed_codes[i / 2] >> 4) & 0x07);
            out[i] = centroids_8[code] * norm;
        }
    }
};

// Tier 3: 1-bit QJL Long-range State (Popcount-searchable)
struct QJLLongState {
    float norm;
    std::vector<uint64_t> bit_mask; // 64 signs per uint64_t
    
    // Fast bitwise dot-product similarity via Hamming distance / POPCNT
    int hamming_distance(const QJLLongState& other) const {
        int dist = 0;
        for (size_t i = 0; i < bit_mask.size(); ++i) {
            #if defined(__x86_64__) || defined(_M_X64)
            dist += _mm_popcnt_u64(bit_mask[i] ^ other.bit_mask[i]);
            #else
            dist += __builtin_popcountll(bit_mask[i] ^ other.bit_mask[i]);
            #endif
        }
        return dist;
    }
};

class HierarchicalMemoryPlacer {
public:
    uint32_t d_state;
    uint32_t d_model;
    
    // Centroids for 3-bit TurboQuant
    std::vector<float> centroids_3bit = {
        -2.15224f, -1.34393f, -0.75601f, -0.24508f,
         0.24508f,  0.75601f,  1.34393f,  2.15224f
    };

    // Tier 1: L1 Cache Ring Buffer (Raw FP32)
    static constexpr size_t L1_CAPACITY = 32;
    std::vector<std::vector<float>> l1_states;
    std::vector<std::vector<float>> l1_keys;

    // Tier 2: L2/L3 Cache (TurboQuant 3-bit)
    static constexpr size_t L3_CAPACITY = 2048;
    std::vector<TurboQuant3State> l3_states;
    std::vector<TurboQuant3State> l3_keys;

    // Tier 3: DRAM Archive (1-bit QJL)
    std::vector<QJLLongState> dram_states;
    std::vector<QJLLongState> dram_keys;

    void init(uint32_t ds, uint32_t dm) {
        d_state = ds;
        d_model = dm;
        float scale = 1.0f / std::sqrt(static_cast<float>(d_state));
        for (auto& c : centroids_3bit) c *= scale;

        l1_states.clear();
        l1_keys.clear();
        l3_states.clear();
        l3_keys.clear();
        dram_states.clear();
        dram_keys.clear();
    }

    /**
     * Stores a new memory checkpoint with automatic hierarchical tier migration.
     */
    void add_checkpoint(const float* state, const float* key) {
        // 1. Store into L1 Tier
        std::vector<float> st(state, state + d_state);
        std::vector<float> ky(key, key + d_model);
        l1_states.push_back(st);
        l1_keys.push_back(ky);

        // 2. If L1 overflows, migrate oldest to Tier 2 (L3 TurboQuant 3-bit)
        if (l1_states.size() > L1_CAPACITY) {
            auto oldest_s = l1_states.front();
            auto oldest_k = l1_keys.front();
            l1_states.erase(l1_states.begin());
            l1_keys.erase(l1_keys.begin());

            // Compress to TurboQuant 3-bit
            TurboQuant3State tq_s = quantize_tq3(oldest_s.data(), d_state);
            TurboQuant3State tq_k = quantize_tq3(oldest_k.data(), d_model);
            l3_states.push_back(tq_s);
            l3_keys.push_back(tq_k);

            // 3. If L3 overflows, migrate oldest to Tier 3 (DRAM 1-bit QJL)
            if (l3_states.size() > L3_CAPACITY) {
                // Dequantize then QJL pack
                std::vector<float> deq_s(d_state);
                l3_states.front().dequantize(d_state, deq_s.data(), centroids_3bit.data());
                l3_states.erase(l3_states.begin());
                l3_keys.erase(l3_keys.begin());

                QJLLongState qjl_s = quantize_qjl(deq_s.data(), d_state);
                dram_states.push_back(qjl_s);
            }
        }
    }

private:
    TurboQuant3State quantize_tq3(const float* vec, uint32_t dim) {
        TurboQuant3State res;
        float sum_sq = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) sum_sq += vec[i] * vec[i];
        res.norm = std::sqrt(sum_sq) + 1e-8f;

        res.packed_codes.resize((dim + 1) / 2, 0);
        for (uint32_t i = 0; i < dim; ++i) {
            float norm_val = vec[i] / res.norm;
            // Find closest centroid
            uint8_t best_code = 0;
            float min_dist = 1e9f;
            for (uint8_t c = 0; c < 8; ++c) {
                float dist = std::abs(norm_val - centroids_3bit[c]);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_code = c;
                }
            }
            if (i % 2 == 0) {
                res.packed_codes[i / 2] |= (best_code & 0x07);
            } else {
                res.packed_codes[i / 2] |= ((best_code & 0x07) << 4);
            }
        }
        return res;
    }

    QJLLongState quantize_qjl(const float* vec, uint32_t dim) {
        QJLLongState res;
        float sum_sq = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) sum_sq += vec[i] * vec[i];
        res.norm = std::sqrt(sum_sq);

        size_t num_u64 = (dim + 63) / 64;
        res.bit_mask.resize(num_u64, 0ULL);
        for (uint32_t i = 0; i < dim; ++i) {
            if (vec[i] >= 0.0f) {
                res.bit_mask[i / 64] |= (1ULL << (i % 64));
            }
        }
        return res;
    }
};

} // namespace cache
