"""
Bit-MC-SSM (1.58-bit Memory-Cached State Space Model)
Phase 3: 2-bit Binary Exporter (PyTorch -> model.bin)
"""

import struct
import argparse
from typing import Tuple
import numpy as np
import torch

MAGIC_HEADER = 0x42495453  # 'BITS' (Bit-SSM format magic number)
VERSION = 2  # v2: embedding table is INT8-quantized (per-row scale) instead of raw float32


def quantize_embedding_int8(emb_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-row (per-token) symmetric INT8 quantization of the embedding table.
    Ternary (2-bit) quantization is too coarse for embeddings, since each row
    must stay distinguishable from every other token's row; INT8 keeps the
    table's memory footprint small (1/4 of float32) without collapsing that
    distinctiveness, which matters most for memory-constrained targets (e.g.
    ESP32-S3) where a large vocab's float32 embedding table would otherwise
    dominate RAM usage on its own.

    Returns:
      scales (float32 array, shape [vocab_size]): per-row dequantization scale
      packed (int8 array, shape [vocab_size, d_model]): quantized values
    """
    w = emb_tensor.detach().cpu().float().numpy()
    abs_max = np.clip(np.abs(w).max(axis=1, keepdims=True), 1e-8, None)
    scales = (abs_max / 127.0).astype(np.float32)
    w_int8 = np.clip(np.round(w / scales), -127, 127).astype(np.int8)
    return scales.flatten(), w_int8

def pack_ternary_weights(w_tensor: torch.Tensor) -> Tuple[float, bytes]:
    """
    Quantizes a 2D weight tensor [out_features, in_features] to ternary {-1, 0, +1}
    and packs 4 weights into 1 byte (2 bits per weight).
    Encoding:
      0  -> 0b00 (0)
      +1 -> 0b01 (1)
      -1 -> 0b10 (2)
    Returns:
      gamma (float): scaling factor
      packed_bytes (bytes): packed binary buffer
    """
    w = w_tensor.detach().cpu().float().numpy()
    gamma = float(np.mean(np.abs(w)) + 1e-5)
    w_scaled = w / gamma
    w_ternary = np.clip(np.round(w_scaled), -1.0, 1.0).astype(np.int8)

    flat = w_ternary.flatten()
    total_weights = len(flat)

    # Pad to multiple of 4
    pad_len = (4 - (total_weights % 4)) % 4
    if pad_len > 0:
        flat = np.pad(flat, (0, pad_len), mode="constant", constant_values=0)

    # Map: 0 -> 0, 1 -> 1, -1 -> 2
    mapped = np.zeros_like(flat, dtype=np.uint8)
    mapped[flat == 0] = 0
    mapped[flat == 1] = 1
    mapped[flat == -1] = 2

    # Pack 4 values per byte: (v0) | (v1 << 2) | (v2 << 4) | (v3 << 6)
    mapped_reshaped = mapped.reshape(-1, 4)
    packed = (
        (mapped_reshaped[:, 0] & 0x03) |
        ((mapped_reshaped[:, 1] & 0x03) << 2) |
        ((mapped_reshaped[:, 2] & 0x03) << 4) |
        ((mapped_reshaped[:, 3] & 0x03) << 6)
    ).astype(np.uint8)

    return gamma, packed.tobytes()

def export_checkpoint(checkpoint_path: str, output_bin: str, vocab_file: str = "vocab.txt"):
    print(f"📦 Loading PyTorch checkpoint: {checkpoint_path}")
    data = torch.load(checkpoint_path, map_location="cpu")
    state_dict = data["model_state_dict"]
    config = data["config"]

    vocab_size = int(config.get("vocab_size", 50257))
    d_model = int(config.get("d_model", 128))
    n_layers = int(config.get("n_layers", 4))
    d_state = int(config.get("d_state", 16))
    segment_len = int(config.get("segment_len", 32))
    top_k = int(config.get("top_k", 2))
    conv_kernel = 4

    print("=" * 60)
    print("⚙️ Model Architecture Config:")
    print(f"   Vocab Size: {vocab_size} | d_model: {d_model} | Layers: {n_layers}")
    print(f"   d_state: {d_state} | Segment Len (C): {segment_len} | Top-k: {top_k}")
    print("=" * 60)

    with open(output_bin, "wb") as f:
        # 1. Header (Magic, Version, Hyperparameters)
        # format: <IIIIIIII (8 unsigned 32-bit ints)
        header_bytes = struct.pack(
            "<IIIIIIII",
            MAGIC_HEADER,
            VERSION,
            vocab_size,
            d_model,
            n_layers,
            d_state,
            segment_len,
            top_k
        )
        f.write(header_bytes)

        # 2. Embedding Table [vocab_size, d_model], INT8-quantized (v2+):
        #    [vocab_size] float32 per-row scales, then [vocab_size * d_model] int8 values
        emb_scales, emb_q8 = quantize_embedding_int8(state_dict["embedding.weight"])
        f.write(emb_scales.tobytes())
        f.write(emb_q8.tobytes())

        # 3. Layer by Layer Export
        for l in range(n_layers):
            prefix = f"blocks.{l}."

            # Layer Norm 1
            w_norm1 = state_dict[f"{prefix}norm1.weight"].detach().cpu().float().numpy()
            f.write(w_norm1.tobytes())

            # ShiftSSM in_proj [2 * d_model, d_model]
            gamma, packed = pack_ternary_weights(state_dict[f"{prefix}mc_ssm.ssm.in_proj.weight"])
            f.write(struct.pack("<f", gamma))
            f.write(packed)

            # ShiftSSM 1D Conv [d_model, 1, conv_kernel] & bias [d_model]
            conv_w = state_dict[f"{prefix}mc_ssm.ssm.conv1d.weight"].detach().cpu().float().numpy().flatten()
            conv_b = state_dict[f"{prefix}mc_ssm.ssm.conv1d.bias"].detach().cpu().float().numpy().flatten()
            f.write(conv_w.tobytes())
            f.write(conv_b.tobytes())

            # ShiftSSM B proj [d_state, d_model]
            gamma, packed = pack_ternary_weights(state_dict[f"{prefix}mc_ssm.ssm.b_proj.weight"])
            f.write(struct.pack("<f", gamma))
            f.write(packed)

            # ShiftSSM C proj [d_state, d_model]
            gamma, packed = pack_ternary_weights(state_dict[f"{prefix}mc_ssm.ssm.c_proj.weight"])
            f.write(struct.pack("<f", gamma))
            f.write(packed)

            # ShiftSSM decay factor (sigmoid of decay_param) [d_state]
            decay_p = state_dict[f"{prefix}mc_ssm.ssm.decay_param"].detach().cpu().float()
            decay_val = torch.sigmoid(decay_p).numpy().astype(np.float32)
            f.write(decay_val.tobytes())

            # ShiftSSM out_proj [d_model, d_model]
            gamma, packed = pack_ternary_weights(state_dict[f"{prefix}mc_ssm.ssm.out_proj.weight"])
            f.write(struct.pack("<f", gamma))
            f.write(packed)

            # MC query_proj [d_model, d_model]
            gamma, packed = pack_ternary_weights(state_dict[f"{prefix}mc_ssm.query_proj.weight"])
            f.write(struct.pack("<f", gamma))
            f.write(packed)

            # MC key_proj [d_model, d_model]
            gamma, packed = pack_ternary_weights(state_dict[f"{prefix}mc_ssm.key_proj.weight"])
            f.write(struct.pack("<f", gamma))
            f.write(packed)

            # MC gate_proj [2, d_model]
            gamma, packed = pack_ternary_weights(state_dict[f"{prefix}mc_ssm.gate_proj.weight"])
            f.write(struct.pack("<f", gamma))
            f.write(packed)

            # MC cache_out_proj [d_model, d_state]
            gamma, packed = pack_ternary_weights(state_dict[f"{prefix}mc_ssm.cache_out_proj.weight"])
            f.write(struct.pack("<f", gamma))
            f.write(packed)

            # Layer Norm 2
            w_norm2 = state_dict[f"{prefix}norm2.weight"].detach().cpu().float().numpy()
            f.write(w_norm2.tobytes())

            # FFN in [2 * (d_model * 2), d_model]
            gamma, packed = pack_ternary_weights(state_dict[f"{prefix}ffn_in.weight"])
            f.write(struct.pack("<f", gamma))
            f.write(packed)

            # FFN out [d_model, d_model * 2]
            gamma, packed = pack_ternary_weights(state_dict[f"{prefix}ffn_out.weight"])
            f.write(struct.pack("<f", gamma))
            f.write(packed)

        # 4. Final Norm
        final_norm = state_dict["final_norm.weight"].detach().cpu().float().numpy()
        f.write(final_norm.tobytes())

        # 5. LM Head [vocab_size, d_model]
        gamma, packed = pack_ternary_weights(state_dict["lm_head.weight"])
        f.write(struct.pack("<f", gamma))
        f.write(packed)

    # 6. Export Vocabulary mapping
    try:
        out_dir = os.path.dirname(output_bin) or "."
        vocab_path = os.path.join(out_dir, "vocab.json")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        vocab = tok.get_vocab() # dict token -> id
        # Invert to id -> token string
        id_to_token = {v: k for k, v in vocab.items()}
        import json
        with open(vocab_path, "w", encoding="utf-8") as vf:
            json.dump(id_to_token, vf, ensure_ascii=False)
        print(f"📖 Exported {vocab_path} successfully!")
    except Exception as e:
        print(f"ℹ️ Could not export full vocab.json ({e})")

    # Calculate file sizes
    pt_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    bin_size_mb = os.path.getsize(output_bin) / (1024 * 1024)

    print("\n" + "=" * 60)
    print("🎉 Export Completed Successfully!")
    print(f"   Original PyTorch (.pt) Size: {pt_size_mb:.2f} MB")
    print(f"   Packed 2-bit Binary (.bin) Size: {bin_size_mb:.2f} MB")
    print(f"   Compression Factor: {pt_size_mb / max(bin_size_mb, 1e-4):.2f}x reduction!")
    print(f"   Saved to: {output_bin}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/bit_mc_ssm_phase2.pt", help="Path to .pt file")
    parser.add_argument("--output", type=str, default="models/model.bin", help="Output .bin file")
    args = parser.parse_args()

    import os
    if not os.path.exists(args.checkpoint):
        if os.path.exists("bit_mc_ssm_phase2.pt"):
            args.checkpoint = "bit_mc_ssm_phase2.pt"
        elif os.path.exists("models/bit_mc_ssm_phase1.pt"):
            args.checkpoint = "models/bit_mc_ssm_phase1.pt"
        elif os.path.exists("bit_mc_ssm_phase1.pt"):
            args.checkpoint = "bit_mc_ssm_phase1.pt"

    export_checkpoint(args.checkpoint, args.output)
