"""
Bit-MC-SSM: Unified Training & 2-bit Binary Export Script
Supports both CPU and CUDA with GaLore Low-Rank Optimizer, BitNet v2 (H-BitLinear), and Delta-SSM.
"""

import os
import sys
import time
import math
import argparse
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from python.galore_optimizer import GaLoreAdamW
from python.h_bitlinear import HBitLinear
from python.delta_ssm import DeltaSSMBlock
from python.export_model import pack_ternary_weights


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class BitMCSSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 32, tau: float = 0.85):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.ssm = DeltaSSMBlock(d_model=d_model, d_state=d_state, delta_thresh=0.01)
        self.norm2 = RMSNorm(d_model)
        self.ffn_in = HBitLinear(d_model, d_model * 4, tau=tau, use_hadamard=False)
        self.ffn_out = HBitLinear(d_model * 2, d_model, tau=tau, use_hadamard=True)

    def forward(self, x: torch.Tensor):
        ssm_out, _ = self.ssm.forward_sequence(self.norm1(x))
        x = x + ssm_out
        ffn_p = self.ffn_in(self.norm2(x))
        f1, f2 = ffn_p.chunk(2, dim=-1)
        x = x + self.ffn_out(F.silu(f1) * f2)
        return x


class BitMCSSM(nn.Module):
    def __init__(self, vocab_size: int = 50257, d_model: int = 384, n_layers: int = 8, d_state: int = 32, tau: float = 0.85):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            BitMCSSMBlock(d_model=d_model, d_state=d_state, tau=tau)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = HBitLinear(d_model, vocab_size, tau=tau, use_hadamard=False)

    def forward(self, idx: torch.Tensor):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)


class SyntheticStoryDataset(Dataset):
    def __init__(self, num_samples: int = 1000, seq_len: int = 64, vocab_size: int = 50257):
        self.data = []
        patterns = [
            [7454, 2402, 257, 640, 11, 20037, 2497, 257, 44152, 1757, 393, 20037, 547, 257, 2833, 4037],
            [464, 1310, 3290, 3058, 284, 862, 287, 262, 3867, 13, 6342, 547, 1266, 284, 4015],
            [1881, 10007, 3347, 11, 3867, 1043, 257, 4454, 284, 18512, 13, 383, 1757, 547, 3772]
        ]
        for i in range(num_samples):
            base = patterns[i % len(patterns)]
            repeats = (seq_len // len(base)) + 1
            seq = (base * repeats)[:seq_len]
            self.data.append(torch.tensor(seq, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x[:-1], x[1:]


def export_binary(model: BitMCSSM, out_path: str = "model_medium-30M.bin"):
    print(f"📦 Exporting model to 2-bit binary: {out_path}...")
    model.eval().cpu()
    with open(out_path, "wb") as f:
        # Magic 'BSSM' = 0x4D535342
        header = struct.pack("<IIIII", 0x4D535342, model.vocab_size, model.d_model, model.n_layers, model.d_state)
        f.write(header)

        # Embedding table
        emb_data = model.tok_emb.weight.detach().float().numpy().tobytes()
        f.write(emb_data)

        # Layers
        for block in model.blocks:
            # norm1
            f.write(block.norm1.weight.detach().float().numpy().tobytes())

            # ssm
            ssm = block.ssm
            # in_proj
            g_in, p_in = pack_ternary_weights(ssm.in_proj.weight)
            f.write(struct.pack("<f", g_in) + p_in)
            # conv1d
            f.write(ssm.conv1d.weight.detach().float().numpy().tobytes())
            f.write(ssm.conv1d.bias.detach().float().numpy().tobytes())
            # b_proj & c_proj
            g_b, p_b = pack_ternary_weights(ssm.b_proj.weight)
            f.write(struct.pack("<f", g_b) + p_b)
            g_c, p_c = pack_ternary_weights(ssm.c_proj.weight)
            f.write(struct.pack("<f", g_c) + p_c)
            # decay (fast)
            f.write(ssm.decay_fast.detach().float().numpy().tobytes())
            # out_proj
            g_out, p_out = pack_ternary_weights(ssm.out_proj.weight)
            f.write(struct.pack("<f", g_out) + p_out)

            # norm2
            f.write(block.norm2.weight.detach().float().numpy().tobytes())

            # ffn_in & ffn_out
            g_fin, p_fin = pack_ternary_weights(block.ffn_in.weight)
            f.write(struct.pack("<f", g_fin) + p_fin)
            g_fout, p_fout = pack_ternary_weights(block.ffn_out.weight)
            f.write(struct.pack("<f", g_fout) + p_fout)

        # Final norm & head
        f.write(model.final_norm.weight.detach().float().numpy().tobytes())
        g_head, p_head = pack_ternary_weights(model.lm_head.weight)
        f.write(struct.pack("<f", g_head) + p_head)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"✅ Exported successfully! Binary size: {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Bit-MC-SSM Training & Export")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--d_model", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--d_state", type=int, default=16, help="SSM state dimension")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocab size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--galore_rank", type=int, default=8, help="GaLore projection rank")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_bin", type=str, default="model.bin", help="Output 2-bit binary path")
    args = parser.parse_args()

    print("======================================================================")
    print(f"⚡ Bit-MC-SSM Training on {args.device.upper()}")
    print(f"   Config: d_model={args.d_model}, layers={args.n_layers}, d_state={args.d_state}")
    print("======================================================================")

    model = BitMCSSM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state
    ).to(args.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Total Model Parameters: {total_params:,}")

    optimizer = GaLoreAdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        rank=args.galore_rank,
        update_proj_gap=50
    )

    dataset = SyntheticStoryDataset(num_samples=1000, seq_len=64, vocab_size=args.vocab_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch:02d}/{args.epochs:02d}]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, args.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            ppl = math.exp(min(loss.item(), 100))
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "ppl": f"{ppl:.2f}"})

    export_binary(model, args.out_bin)


if __name__ == "__main__":
    main()
