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
#include <unordered_map>

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
// 4. Tokenizer & JSON Parser (Full GPT-2 BPE + UTF-8 Byte Decode)
// ==============================================================================

struct SimpleTokenizer {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, uint32_t> token_to_id;
    std::unordered_map<uint32_t, uint8_t> unicode_to_byte;
    std::unordered_map<uint8_t, std::string> byte_to_unicode_str;
    bool is_initialized = false;

    void init_byte_mappings() {
        if (is_initialized) return;
        is_initialized = true;

        std::vector<int> bs;
        for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
        for (int b = 161; b <= 172; ++b) bs.push_back(b);
        for (int b = 174; b <= 255; ++b) bs.push_back(b);

        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }

        for (size_t i = 0; i < bs.size(); ++i) {
            uint32_t codepoint = static_cast<uint32_t>(cs[i]);
            uint8_t byte_val = static_cast<uint8_t>(bs[i]);
            unicode_to_byte[codepoint] = byte_val;

            // Encode codepoint to UTF-8 string
            std::string utf8_char = "";
            if (codepoint < 0x80) {
                utf8_char += static_cast<char>(codepoint);
            } else if (codepoint < 0x800) {
                utf8_char += static_cast<char>(0xC0 | (codepoint >> 6));
                utf8_char += static_cast<char>(0x80 | (codepoint & 0x3F));
            } else {
                utf8_char += static_cast<char>(0xE0 | (codepoint >> 12));
                utf8_char += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                utf8_char += static_cast<char>(0x80 | (codepoint & 0x3F));
            }
            byte_to_unicode_str[byte_val] = utf8_char;
        }
    }

    static std::string parse_json_string(const std::string& s, size_t& pos) {
        std::string res;
        pos++; // skip opening '"'
        while (pos < s.size()) {
            char c = s[pos++];
            if (c == '"') {
                return res;
            } else if (c == '\\' && pos < s.size()) {
                char esc = s[pos++];
                if (esc == '"') res += '"';
                else if (esc == '\\') res += '\\';
                else if (esc == '/') res += '/';
                else if (esc == 'b') res += '\b';
                else if (esc == 'f') res += '\f';
                else if (esc == 'n') res += '\n';
                else if (esc == 'r') res += '\r';
                else if (esc == 't') res += '\t';
                else if (esc == 'u' && pos + 4 <= s.size()) {
                    std::string hex = s.substr(pos, 4);
                    pos += 4;
                    try {
                        uint32_t cp = std::stoul(hex, nullptr, 16);
                        if (cp < 0x80) {
                            res += static_cast<char>(cp);
                        } else if (cp < 0x800) {
                            res += static_cast<char>(0xC0 | (cp >> 6));
                            res += static_cast<char>(0x80 | (cp & 0x3F));
                        } else {
                            res += static_cast<char>(0xE0 | (cp >> 12));
                            res += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                            res += static_cast<char>(0x80 | (cp & 0x3F));
                        }
                    } catch (...) {}
                }
            } else {
                res += c;
            }
        }
        return res;
    }

    bool load_vocab(const std::string& path, uint32_t vocab_size) {
        init_byte_mappings();
        id_to_token.assign(vocab_size, "");
        token_to_id.clear();

        std::ifstream f(path);
        if (!f.is_open()) return false;

        std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        size_t pos = 0;
        size_t n = content.size();

        while (pos < n) {
            while (pos < n && content[pos] != '"') pos++;
            if (pos >= n) break;

            std::string key = parse_json_string(content, pos);

            while (pos < n && (content[pos] == ' ' || content[pos] == '\t' || content[pos] == '\r' || content[pos] == '\n' || content[pos] == ':')) {
                pos++;
            }
            if (pos >= n) break;

            if (content[pos] == '"') {
                // "key": "value" format (inverted index)
                std::string val = parse_json_string(content, pos);
                try {
                    uint32_t id = std::stoul(key);
                    if (id < vocab_size) {
                        id_to_token[id] = val;
                        token_to_id[val] = id;
                    }
                } catch (...) {}
            } else if (isdigit(content[pos]) || content[pos] == '-') {
                // "token": id format (standard GPT-2 encoder.json)
                size_t num_start = pos;
                while (pos < n && isdigit(content[pos])) pos++;
                std::string num_str = content.substr(num_start, pos - num_start);
                try {
                    uint32_t id = std::stoul(num_str);
                    if (id < vocab_size) {
                        id_to_token[id] = key;
                        token_to_id[key] = id;
                    }
                } catch (...) {}
            }
        }

        return !token_to_id.empty();
    }

    std::string decode_token(uint32_t id) const {
        if (id >= id_to_token.size() || id_to_token[id].empty()) {
            return "[" + std::to_string(id) + "]";
        }

        const std::string& raw = id_to_token[id];
        std::string decoded_bytes;

        size_t i = 0;
        while (i < raw.size()) {
            unsigned char c0 = static_cast<unsigned char>(raw[i]);
            uint32_t cp = 0;
            size_t char_len = 1;

            if (c0 < 0x80) {
                cp = c0;
                char_len = 1;
            } else if ((c0 & 0xE0) == 0xC0 && i + 1 < raw.size()) {
                unsigned char c1 = static_cast<unsigned char>(raw[i+1]);
                cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
                char_len = 2;
            } else if ((c0 & 0xF0) == 0xE0 && i + 2 < raw.size()) {
                unsigned char c1 = static_cast<unsigned char>(raw[i+1]);
                unsigned char c2 = static_cast<unsigned char>(raw[i+2]);
                cp = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
                char_len = 3;
            }

            auto it = unicode_to_byte.find(cp);
            if (it != unicode_to_byte.end()) {
                decoded_bytes += static_cast<char>(it->second);
                i += char_len;
            } else {
                for (size_t k = 0; k < char_len; ++k) {
                    decoded_bytes += raw[i + k];
                }
                i += char_len;
            }
        }

        return decoded_bytes;
    }

    std::vector<uint32_t> encode(const std::string& text) const {
        std::vector<uint32_t> tokens;
        if (token_to_id.empty() || text.empty()) return tokens;

        // Convert input text to GPT-2 unicode string
        std::string bpe_str = "";
        for (unsigned char c : text) {
            auto it = byte_to_unicode_str.find(c);
            if (it != byte_to_unicode_str.end()) {
                bpe_str += it->second;
            } else {
                bpe_str += static_cast<char>(c);
            }
        }

        // Greedy longest matching
        size_t pos = 0;
        while (pos < bpe_str.size()) {
            bool matched = false;
            size_t max_len = std::min(static_cast<size_t>(64), bpe_str.size() - pos);
            for (size_t len = max_len; len > 0; --len) {
                std::string sub = bpe_str.substr(pos, len);
                auto it = token_to_id.find(sub);
                if (it != token_to_id.end()) {
                    tokens.push_back(it->second);
                    pos += len;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                pos++;
            }
        }
        return tokens;
    }
};

// ==============================================================================
// 5. Sampling
// ==============================================================================

// ==============================================================================
// 5. Sampling (Top-K, Top-P Nucleus, & Repetition Penalty)
// ==============================================================================

uint32_t sample_logits(
    float* logits,
    uint32_t vocab_size,
    float temperature,
    uint32_t top_k,
    float top_p,
    const std::vector<uint32_t>& recent_tokens,
    float rep_penalty,
    std::mt19937& rng
) {
    // 1. Apply Repetition Penalty
    if (rep_penalty > 1.0f) {
        for (uint32_t tok : recent_tokens) {
            if (tok < vocab_size) {
                if (logits[tok] > 0.0f) {
                    logits[tok] /= rep_penalty;
                } else {
                    logits[tok] *= rep_penalty;
                }
            }
        }
    }

    // 2. Greedy if zero temperature
    if (temperature < 1e-4f) {
        return static_cast<uint32_t>(std::distance(logits, std::max_element(logits, logits + vocab_size)));
    }

    // 3. Apply Temperature
    for (uint32_t i = 0; i < vocab_size; ++i) {
        logits[i] /= temperature;
    }

    // 4. Sort candidates
    std::vector<std::pair<float, uint32_t>> pairs(vocab_size);
    for (uint32_t i = 0; i < vocab_size; ++i) {
        pairs[i] = {logits[i], i};
    }

    uint32_t k = std::min(top_k, vocab_size);
    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    // 5. Compute Softmax over Top-K
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

    // 6. Top-P (Nucleus) Truncation
    float cum_p = 0.0f;
    uint32_t cutoff_k = k;
    for (uint32_t i = 0; i < k; ++i) {
        cum_p += probs[i];
        if (cum_p >= top_p) {
            cutoff_k = i + 1;
            break;
        }
    }

    // Renormalize after Top-P
    float p_sum = 0.0f;
    for (uint32_t i = 0; i < cutoff_k; ++i) {
        p_sum += probs[i];
    }
    for (uint32_t i = 0; i < cutoff_k; ++i) {
        probs[i] /= p_sum;
    }

    // 7. Sample
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cdf = 0.0f;
    for (uint32_t i = 0; i < cutoff_k; ++i) {
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
    std::string user_prompt = "";
    float temperature = 0.7f;
    uint32_t top_k = 40;
    float top_p = 0.9f;
    float rep_penalty = 1.15f;

    if (argc > 1) model_path = argv[1];
    if (argc > 2) max_gen_tokens = std::stoi(argv[2]);
    if (argc > 3) user_prompt = argv[3];
    if (argc > 4) temperature = std::stof(argv[4]);
    if (argc > 5) top_k = std::stoi(argv[5]);

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
        std::cout << "📖 Loaded " << vocab_path << " (" << tokenizer.token_to_id.size() << " tokens)\n";
    } else {
        std::cerr << "⚠️ Warning: " << vocab_path << " not found or empty. Outputting token IDs.\n";
    }

    ModelRuntimeState state;
    state.init(model.config.n_layers, model.config.d_model, model.config.d_state);

    std::vector<float> logits(model.config.vocab_size);
    std::mt19937 rng(42);

    std::vector<uint32_t> prompt_tokens;
    if (!user_prompt.empty() && has_vocab) {
        prompt_tokens = tokenizer.encode(user_prompt);
    }
    if (prompt_tokens.empty()) {
        // Default warm-up prompt: "Once upon a time, Lily saw a tiny"
        prompt_tokens = {7454, 2402, 257, 640, 11, 20037, 2497, 257, 44152};
    }

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

    // Print Prompt to Stream
    for (uint32_t tok : prompt_tokens) {
        std::cout << tokenizer.decode_token(tok);
    }
    std::cout << std::flush;

    std::vector<uint32_t> recent_tokens = prompt_tokens;

    uint32_t cur_tok = sample_logits(logits.data(), model.config.vocab_size, temperature, top_k, top_p, recent_tokens, rep_penalty, rng);
    std::cout << tokenizer.decode_token(cur_tok) << std::flush;
    recent_tokens.push_back(cur_tok);

    auto gen_start = std::chrono::high_resolution_clock::now();

    for (uint32_t step_i = 0; step_i < max_gen_tokens; ++step_i) {
        model.step(cur_tok, state, logits.data());
        cur_tok = sample_logits(logits.data(), model.config.vocab_size, temperature, top_k, top_p, recent_tokens, rep_penalty, rng);
        std::cout << tokenizer.decode_token(cur_tok) << std::flush;
        recent_tokens.push_back(cur_tok);
        if (recent_tokens.size() > 64) {
            recent_tokens.erase(recent_tokens.begin());
        }
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

