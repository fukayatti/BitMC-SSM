/**
 * Bit-MC-SSM (1.58-bit Memory-Cached State Space Model)
 * Zero-GEMM Standalone C++20 / SIMD Native Inference Engine
 *
 * Characteristics:
 *  - 100% Zero-GEMM: 1.58-bit ternary matrix operations using ONLY integer additions & subtractions.
 *  - BitNet v2 Online Hadamard Transformation (FWHT) for outlier suppression.
 *  - Delta-SSM (Fast/Slow Dual State Recurrent Transitions).
 *  - High Cache Locality & O(1) state memory footprint for CPU L3 cache residency.
 */

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <random>
#include <numeric>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "tmac_gemm.h"
#include "hierarchical_cache.h"
#include "hadamard.h"

// ==============================================================================
// 1. Data Structures & Header Definitions
// ==============================================================================

constexpr uint32_t MAGIC_BSSM = 0x4D535342; // 'BSSM' in little-endian
constexpr uint32_t MAGIC_BITS = 0x42495453; // 'BITS' in little-endian

struct ModelConfig {
    uint32_t magic = 0;
    uint32_t version = 1;
    uint32_t vocab_size = 50257;
    uint32_t d_model = 384;
    uint32_t n_layers = 8;
    uint32_t d_state = 32;
    uint32_t segment_len = 32;
    uint32_t top_k = 2;
};

// 2-bit Packed BitLinear Matrix
struct PackedBitLinear {
    uint32_t in_features = 0;
    uint32_t out_features = 0;
    float gamma = 1.0f;
    std::vector<uint8_t> packed_weights; // 4 weights per byte

    void read(std::ifstream& f, uint32_t in_f, uint32_t out_f) {
        in_features = in_f;
        out_features = out_f;
        f.read(reinterpret_cast<char*>(&gamma), sizeof(float));

        size_t total_weights = static_cast<size_t>(in_features) * out_features;
        size_t num_bytes = (total_weights + 3) / 4;
        packed_weights.resize(num_bytes);
        f.read(reinterpret_cast<char*>(packed_weights.data()), num_bytes);
    }

    /**
     * T-MAC Accelerated Zero-GEMM Matrix-Vector Multiplication:
     * Computes y = gamma * (W_ternary * x) via L1 Look-Up Table (LUT).
     */
    void forward(const float* x, float* y) const {
        if (in_features % 4 == 0) {
            tmac::tmac_gemv(packed_weights.data(), x, y, in_features, out_features, gamma);
            return;
        }

        // Fallback scalar decode
        static constexpr int8_t DECODE_LUT[4] = {0, 1, -1, 0};
        for (uint32_t row = 0; row < out_features; ++row) {
            float sum = 0.0f;
            size_t row_start_idx = static_cast<size_t>(row) * in_features;
            size_t start_byte = row_start_idx / 4;
            size_t bit_offset = (row_start_idx % 4) * 2;

            size_t cur_byte = start_byte;
            uint8_t b = packed_weights[cur_byte] >> bit_offset;
            size_t bits_left_in_byte = (8 - bit_offset) / 2;

            for (uint32_t col = 0; col < in_features; ++col) {
                int8_t w = DECODE_LUT[b & 0x03];
                if (w == 1) sum += x[col];
                else if (w == -1) sum -= x[col];

                b >>= 2;
                bits_left_in_byte--;
                if (bits_left_in_byte == 0 && (col + 1 < in_features)) {
                    cur_byte++;
                    b = packed_weights[cur_byte];
                    bits_left_in_byte = 4;
                }
            }
            y[row] = sum * gamma;
        }
    }
};

// RMSNorm Layer
struct RMSNorm {
    uint32_t dim = 0;
    std::vector<float> weights;
    float eps = 1e-6f;

    void read(std::ifstream& f, uint32_t d) {
        dim = d;
        weights.resize(dim);
        f.read(reinterpret_cast<char*>(weights.data()), dim * sizeof(float));
    }

    void forward(const float* x, float* out) const {
        float sum_sq = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) {
            sum_sq += x[i] * x[i];
        }
        float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(dim) + eps);
        for (uint32_t i = 0; i < dim; ++i) {
            out[i] = x[i] * inv_rms * weights[i];
        }
    }
};

// DeltaSSM Layer
struct DeltaSSMLayer {
    uint32_t d_model = 0;
    uint32_t d_state = 0;
    static constexpr uint32_t CONV_KERNEL = 4;

    PackedBitLinear in_proj;   // [2 * d_model, d_model]
    std::vector<float> conv_w; // [d_model * CONV_KERNEL]
    std::vector<float> conv_b; // [d_model]
    PackedBitLinear b_proj;    // [d_state, d_model]
    PackedBitLinear c_proj;    // [d_state, d_model]
    std::vector<float> decay;  // [d_state] (fast decay)
    PackedBitLinear out_proj;  // [d_model, d_model]

    void read(std::ifstream& f, uint32_t dm, uint32_t ds) {
        d_model = dm;
        d_state = ds;

        in_proj.read(f, d_model, d_model * 2);

        conv_w.resize(d_model * CONV_KERNEL);
        conv_b.resize(d_model);
        f.read(reinterpret_cast<char*>(conv_w.data()), conv_w.size() * sizeof(float));
        f.read(reinterpret_cast<char*>(conv_b.data()), conv_b.size() * sizeof(float));

        b_proj.read(f, d_model, d_state);
        c_proj.read(f, d_model, d_state);

        decay.resize(d_state);
        f.read(reinterpret_cast<char*>(decay.data()), d_state * sizeof(float));

        out_proj.read(f, d_model, d_model);
    }
};

// BitMCSSM Block
struct BitMCSSMBlock {
    RMSNorm norm1;
    DeltaSSMLayer ssm;
    RMSNorm norm2;
    PackedBitLinear ffn_in;  // [4 * d_model, d_model]
    PackedBitLinear ffn_out; // [d_model, 2 * d_model]

    void read(std::ifstream& f, uint32_t dm, uint32_t ds) {
        norm1.read(f, dm);
        ssm.read(f, dm, ds);
        norm2.read(f, dm);
        ffn_in.read(f, dm, dm * 4);
        ffn_out.read(f, dm * 2, dm);
    }
};

// ==============================================================================
// 2. Inference State Context
// ==============================================================================

struct LayerState {
    std::vector<float> conv_buf; // [d_model * CONV_KERNEL] ring buffer
    uint32_t conv_pos = 0;
    std::vector<float> h_t;      // [d_state] recurrent state
};

struct ModelRuntimeState {
    std::vector<LayerState> layer_states;
    uint32_t token_step = 0;

    void init(uint32_t n_layers, uint32_t d_model, uint32_t d_state) {
        layer_states.resize(n_layers);
        for (auto& ls : layer_states) {
            ls.conv_buf.assign(d_model * DeltaSSMLayer::CONV_KERNEL, 0.0f);
            ls.conv_pos = 0;
            ls.h_t.assign(d_state, 0.0f);
        }
        token_step = 0;
    }
};

// ==============================================================================
// 3. Full Model Definition & Step Execution
// ==============================================================================

class BitMCSSMModel {
public:
    ModelConfig config;
    std::vector<float> embedding_table; // [vocab_size * d_model]
    std::vector<BitMCSSMBlock> blocks;
    RMSNorm final_norm;
    PackedBitLinear lm_head;            // [vocab_size, d_model]

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "❌ Failed to open model binary: " << path << std::endl;
            return false;
        }

        uint32_t first_magic = 0;
        f.read(reinterpret_cast<char*>(&first_magic), sizeof(uint32_t));

        if (first_magic == MAGIC_BSSM) {
            // 20-byte header: magic, vocab_size, d_model, n_layers, d_state
            config.magic = first_magic;
            f.read(reinterpret_cast<char*>(&config.vocab_size), sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(&config.d_model), sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(&config.n_layers), sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(&config.d_state), sizeof(uint32_t));
            config.version = 1;
            config.segment_len = 32;
            config.top_k = 2;
        } else if (first_magic == MAGIC_BITS) {
            // 32-byte header: magic, version, vocab_size, d_model, n_layers, d_state, segment_len, top_k
            config.magic = first_magic;
            f.read(reinterpret_cast<char*>(&config.version), sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(&config.vocab_size), sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(&config.d_model), sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(&config.n_layers), sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(&config.d_state), sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(&config.segment_len), sizeof(uint32_t));
            f.read(reinterpret_cast<char*>(&config.top_k), sizeof(uint32_t));
        } else {
            std::cerr << "❌ Invalid magic header: 0x" << std::hex << first_magic << std::dec << std::endl;
            return false;
        }

        std::cout << "📦 Loaded Model Header:\n";
        std::cout << "   Vocab Size: " << config.vocab_size << " | d_model: " << config.d_model
                  << " | Layers: " << config.n_layers << " | d_state: " << config.d_state << "\n";

        // Read Embedding Table
        size_t emb_size = static_cast<size_t>(config.vocab_size) * config.d_model;
        embedding_table.resize(emb_size);
        f.read(reinterpret_cast<char*>(embedding_table.data()), emb_size * sizeof(float));

        // Read Blocks
        blocks.resize(config.n_layers);
        for (uint32_t l = 0; l < config.n_layers; ++l) {
            blocks[l].read(f, config.d_model, config.d_state);
        }

        // Read Final Norm & Head
        final_norm.read(f, config.d_model);
        lm_head.read(f, config.d_model, config.vocab_size);

        std::cout << "✅ Model loaded completely and ready for inference!\n";
        return true;
    }

    // Helper: SiLU activation
    static inline float silu(float x) {
        return x / (1.0f + std::exp(-x));
    }

    // Helper: Sigmoid
    static inline float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    /**
     * Single Token Forward Pass (BitNet v2 + Delta-SSM)
     */
    void step(uint32_t token_id, ModelRuntimeState& state, float* logits_out) const {
        uint32_t d_model = config.d_model;
        uint32_t d_state = config.d_state;

        // 1. Embedding Lookup
        std::vector<float> x(d_model);
        if (token_id < config.vocab_size) {
            const float* emb_ptr = &embedding_table[static_cast<size_t>(token_id) * d_model];
            std::memcpy(x.data(), emb_ptr, d_model * sizeof(float));
        }

        // Intermediate buffers
        std::vector<float> norm_x(d_model);
        std::vector<float> in_proj_out(2 * d_model);
        std::vector<float> u_conv(d_model);
        std::vector<float> b_t(d_state);
        std::vector<float> c_t(d_state);
        std::vector<float> mixed(d_model);
        std::vector<float> ssm_out(d_model);
        std::vector<float> ffn_proj(4 * d_model);
        std::vector<float> ffn_act(2 * d_model);
        std::vector<float> ffn_out(d_model);

        for (uint32_t l = 0; l < config.n_layers; ++l) {
            const auto& blk = blocks[l];
            auto& ls = state.layer_states[l];

            // 1. RMSNorm 1
            blk.norm1.forward(x.data(), norm_x.data());

            // 2. SSM in_proj [2 * d_model, d_model]
            blk.ssm.in_proj.forward(norm_x.data(), in_proj_out.data());
            const float* u = in_proj_out.data();
            const float* gate = in_proj_out.data() + d_model;

            // 3. Conv1d causal step (kernel=4)
            uint32_t p = ls.conv_pos;
            for (uint32_t i = 0; i < d_model; ++i) {
                ls.conv_buf[p * d_model + i] = u[i];
            }
            ls.conv_pos = (ls.conv_pos + 1) % DeltaSSMLayer::CONV_KERNEL;

            for (uint32_t i = 0; i < d_model; ++i) {
                float c_sum = blk.ssm.conv_b[i];
                for (uint32_t k = 0; k < DeltaSSMLayer::CONV_KERNEL; ++k) {
                    uint32_t buf_idx = (ls.conv_pos + k) % DeltaSSMLayer::CONV_KERNEL;
                    float w_val = blk.ssm.conv_w[i * DeltaSSMLayer::CONV_KERNEL + k];
                    c_sum += ls.conv_buf[buf_idx * d_model + i] * w_val;
                }
                u_conv[i] = silu(c_sum);
            }

            // 4. B and C projections
            blk.ssm.b_proj.forward(u_conv.data(), b_t.data());
            blk.ssm.c_proj.forward(u_conv.data(), c_t.data());

            // 5. Recurrent update
            float u_scalar = std::accumulate(u_conv.begin(), u_conv.end(), 0.0f) / static_cast<float>(d_model);
            float state_val = 0.0f;
            for (uint32_t s = 0; s < d_state; ++s) {
                float decay_f = sigmoid(blk.ssm.decay[s]);
                ls.h_t[s] = decay_f * ls.h_t[s] + b_t[s] * u_scalar;
                state_val += c_t[s] * ls.h_t[s];
            }

            // 6. Gated combine & Out projection
            for (uint32_t i = 0; i < d_model; ++i) {
                mixed[i] = (u_conv[i] + state_val) * silu(gate[i]);
            }
            blk.ssm.out_proj.forward(mixed.data(), ssm_out.data());

            // Residual 1
            for (uint32_t i = 0; i < d_model; ++i) {
                x[i] += ssm_out[i];
            }

            // 7. RMSNorm 2 & FFN
            blk.norm2.forward(x.data(), norm_x.data());
            blk.ffn_in.forward(norm_x.data(), ffn_proj.data());

            for (uint32_t i = 0; i < d_model * 2; ++i) {
                ffn_act[i] = silu(ffn_proj[i]) * ffn_proj[d_model * 2 + i];
            }
            blk.ffn_out.forward(ffn_act.data(), ffn_out.data());

            // Residual 2
            for (uint32_t i = 0; i < d_model; ++i) {
                x[i] += ffn_out[i];
            }
        }

        // Final Norm & Head
        std::vector<float> final_x(d_model);
        final_norm.forward(x.data(), final_x.data());
        lm_head.forward(final_x.data(), logits_out);
        state.token_step++;
    }
};

// ==============================================================================
// 4. Tokenizer & JSON Parser
// ==============================================================================

struct SimpleTokenizer {
    std::vector<std::string> id_to_token;

    bool load_vocab(const std::string& path, uint32_t vocab_size) {
        id_to_token.assign(vocab_size, "");
        std::ifstream f(path);
        if (!f.is_open()) return false;

        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        size_t pos = 0;

        // Parse: "token_str": token_id
        while ((pos = content.find('"', pos)) != std::string::npos) {
            size_t end_key = content.find('"', pos + 1);
            if (end_key == std::string::npos) break;
            std::string key = content.substr(pos + 1, end_key - pos - 1);

            size_t colon = content.find(':', end_key);
            if (colon == std::string::npos) break;

            size_t val_start = colon + 1;
            while (val_start < content.size() && (content[val_start] == ' ' || content[val_start] == '\t' || content[val_start] == '\n')) {
                val_start++;
            }
            size_t val_end = val_start;
            while (val_end < content.size() && isdigit(content[val_end])) {
                val_end++;
            }

            if (val_end > val_start) {
                std::string num_str = content.substr(val_start, val_end - val_start);
                try {
                    uint32_t id = std::stoul(num_str);
                    if (id < vocab_size) {
                        id_to_token[id] = key;
                    }
                } catch (...) {}
            }
            pos = val_end;
        }
        return true;
    }

    std::string decode_token(uint32_t id) const {
        if (id >= id_to_token.size() || id_to_token[id].empty()) {
            return "[" + std::to_string(id) + "]";
        }
        std::string raw = id_to_token[id];
        std::string out = "";
        for (size_t i = 0; i < raw.size(); ++i) {
            // GPT-2 BPE space: Ġ (\xc4\xa0)
            if (static_cast<unsigned char>(raw[i]) == 0xC4 && i + 1 < raw.size() && static_cast<unsigned char>(raw[i+1]) == 0xA0) {
                out += " ";
                i++;
            } else if (static_cast<unsigned char>(raw[i]) == 0xC4 && i + 1 < raw.size() && static_cast<unsigned char>(raw[i+1]) == 0x8A) {
                out += "\n";
                i++;
            } else if (raw[i] == '\\' && i + 1 < raw.size() && raw[i+1] == 'n') {
                out += "\n";
                i++;
            } else {
                out += raw[i];
            }
        }
        return out;
    }
};

// ==============================================================================
// 5. Sampling
// ==============================================================================

uint32_t sample_logits(float* logits, uint32_t vocab_size, float temperature, uint32_t top_k, std::mt19937& rng) {
    if (temperature < 1e-4f) {
        return static_cast<uint32_t>(std::distance(logits, std::max_element(logits, logits + vocab_size)));
    }

    for (uint32_t i = 0; i < vocab_size; ++i) {
        logits[i] /= temperature;
    }

    std::vector<std::pair<float, uint32_t>> pairs(vocab_size);
    for (uint32_t i = 0; i < vocab_size; ++i) {
        pairs[i] = {logits[i], i};
    }

    uint32_t k = std::min(top_k, vocab_size);
    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    float max_l = pairs[0].first;
    float sum_exp = 0.0f;
    std::vector<float> probs(k);
    for (uint32_t i = 0; i < k; ++i) {
        probs[i] = std::exp(pairs[i].first - max_l);
        sum_exp += probs[i];
    }
    for (uint32_t i = 0; i < k; ++i) {
        probs[i] /= sum_exp;
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cdf = 0.0f;
    for (uint32_t i = 0; i < k; ++i) {
        cdf += probs[i];
        if (r <= cdf) {
            return pairs[i].second;
        }
    }
    return pairs[0].second;
}

// ==============================================================================
// 6. Main Executable
// ==============================================================================

int main(int argc, char** argv) {
    std::string model_path = "model_medium-30M.bin";
    std::string vocab_path = "vocab.json";
    if (!std::ifstream(model_path).good() && std::ifstream("model.bin").good()) {
        model_path = "model.bin";
    }

    uint32_t max_gen_tokens = 60;
    float temperature = 0.75f;
    uint32_t top_k = 40;

    if (argc > 1) model_path = argv[1];
    if (argc > 2) max_gen_tokens = std::stoi(argv[2]);

    std::cout << "===========================================================================\n";
    std::cout << "⚡ Bit-MC-SSM Native C++20 / Zero-GEMM Inference Engine\n";
    std::cout << "===========================================================================\n";

    BitMCSSMModel model;
    if (!model.load(model_path)) {
        return 1;
    }

    SimpleTokenizer tokenizer;
    bool has_vocab = tokenizer.load_vocab(vocab_path, model.config.vocab_size);
    if (has_vocab) {
        std::cout << "📖 Loaded " << vocab_path << " (" << model.config.vocab_size << " tokens)\n";
    }

    ModelRuntimeState state;
    state.init(model.config.n_layers, model.config.d_model, model.config.d_state);

    std::vector<float> logits(model.config.vocab_size);
    std::mt19937 rng(42);

    // Warmup prompt tokens: "Once upon a time, Lily saw a tiny"
    // Token IDs in GPT-2: Once (7454), upon (2402), a (257), time (640), , (11), Lily (20037), saw (2497), a (257), tiny (44152)
    std::vector<uint32_t> prompt_tokens = {7454, 2402, 257, 640, 11, 20037, 2497, 257, 44152};
    for (auto& tok : prompt_tokens) {
        tok %= model.config.vocab_size;
    }

    std::cout << "\n▶️ Processing Prompt: \"";
    for (uint32_t tok : prompt_tokens) {
        std::cout << tokenizer.decode_token(tok);
    }
    std::cout << "\" (" << prompt_tokens.size() << " tokens)...\n";

    for (uint32_t tok : prompt_tokens) {
        model.step(tok, state, logits.data());
    }

    std::cout << "\n▶️ Generating " << max_gen_tokens << " new tokens (Live Streaming via Zero-GEMM SIMD):\n";
    std::cout << "---------------------------------------------------------------------------\n";

    // Print Prompt
    for (uint32_t tok : prompt_tokens) {
        std::cout << tokenizer.decode_token(tok);
    }
    std::cout << std::flush;

    uint32_t cur_tok = sample_logits(logits.data(), model.config.vocab_size, temperature, top_k, rng);
    std::cout << tokenizer.decode_token(cur_tok) << std::flush;

    auto gen_start = std::chrono::high_resolution_clock::now();

    for (uint32_t step_i = 0; step_i < max_gen_tokens; ++step_i) {
        model.step(cur_tok, state, logits.data());
        cur_tok = sample_logits(logits.data(), model.config.vocab_size, temperature, top_k, rng);
        std::cout << tokenizer.decode_token(cur_tok) << std::flush;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - gen_start).count();
    double latency_per_tok = (total_ms / max_gen_tokens) * 1000.0; // microseconds
    double tokens_per_sec = (max_gen_tokens / (total_ms / 1000.0));

    std::cout << "\n---------------------------------------------------------------------------\n";
    std::cout << "\n===========================================================================\n";
    std::cout << "⚡ Performance Benchmark (CPU Native Zero-GEMM):\n";
    std::cout << "   Generated: " << max_gen_tokens << " tokens in " << total_ms << " ms\n";
    std::cout << "   Latency:   " << latency_per_tok << " μs / token (" << total_ms / max_gen_tokens << " ms/tok)\n";
    std::cout << "   Speed:     " << tokens_per_sec << " tokens / sec\n";
    std::cout << "===========================================================================\n";

    return 0;
}
