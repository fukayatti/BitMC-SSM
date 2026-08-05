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
from python.export_model import pack_ternary_weights


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class DeltaSSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 32, tau: float = 0.85):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = HBitLinear(d_model, 2 * d_model, tau=tau, use_hadamard=False)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        self.b_proj = HBitLinear(d_model, d_state, tau=tau, use_hadamard=False)
        self.c_proj = HBitLinear(d_model, d_state, tau=tau, use_hadamard=False)
        self.decay_fast = nn.Parameter(torch.tensor([-2.0] * d_state))
        self.out_proj = HBitLinear(d_model, d_model, tau=tau, use_hadamard=True)

    def forward(self, x: torch.Tensor, cached_state=None):
        B, L, D = x.shape
        proj = self.in_proj(x)
        u, gate = proj.chunk(2, dim=-1)

        u_conv = self.conv1d(u.transpose(1, 2))[:, :, :L].transpose(1, 2)
        u_conv = F.silu(u_conv)

        B_t = self.b_proj(u_conv)
        C_t = self.c_proj(u_conv)
        decay = torch.sigmoid(self.decay_fast)

        h = cached_state if cached_state is not None else torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        y_list = []
        u_scalar = u_conv.mean(dim=-1, keepdim=True)
        for t in range(L):
            h = decay * h + B_t[:, t, :] * u_scalar[:, t, :]
            y_t = (C_t[:, t, :] * h).sum(dim=-1, keepdim=True)
            y_list.append(y_t)

        y_ssm = torch.cat(y_list, dim=-1).unsqueeze(-1).expand(-1, -1, D)
        y = (u_conv + y_ssm) * F.silu(gate)
        return self.out_proj(y), h


class BitMCSSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 32, tau: float = 0.85):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.ssm = DeltaSSMBlock(d_model=d_model, d_state=d_state, tau=tau)
        self.norm2 = RMSNorm(d_model)
        self.ffn_in = HBitLinear(d_model, d_model * 4, tau=tau, use_hadamard=False)
        self.ffn_out = HBitLinear(d_model * 2, d_model, tau=tau, use_hadamard=True)

    def forward(self, x: torch.Tensor, cached_state=None):
        ssm_out, next_state = self.ssm(self.norm1(x), cached_state)
        x = x + ssm_out
        ffn_p = self.ffn_in(self.norm2(x))
        f1, f2 = ffn_p.chunk(2, dim=-1)
        x = x + self.ffn_out(F.silu(f1) * f2)
        return x, next_state


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
            x, _ = block(x)
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
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--d_model", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--d_state", type=int, default=16, help="SSM state dimension")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocab size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--galore_rank", type=int, default=8, help="GaLore projection rank")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", default=torch.cuda.is_available(), help="Enable Mixed Precision (BF16/FP16)")
    parser.add_argument("--compile", action="store_true", default=False, help="Enable torch.compile acceleration")
    parser.add_argument("--out_bin", type=str, default="model.bin", help="Output 2-bit binary path")
    args = parser.parse_args()

    # Multi-GPU / Distributed Setup (torchrun / DDP)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1

    if is_distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        device = f"cuda:{local_rank}"
    else:
        rank = 0
        world_size = 1
        device = args.device

    is_master = (rank == 0)

    if is_master:
        print("======================================================================")
        print(f"⚡ Bit-MC-SSM Training on {world_size}x GPU(s) (DDP={is_distributed})")
        print(f"   Config: d_model={args.d_model}, layers={args.n_layers}, d_state={args.d_state}")
        print(f"   Optimizations: AMP={args.amp}, compile={args.compile}, grad_accum={args.grad_accum_steps}")
        print(f"   Effective Batch Size: {args.batch_size * args.grad_accum_steps * world_size}")
        print("======================================================================")

    model = BitMCSSM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    if is_master:
        print(f"🧠 Total Model Parameters: {total_params:,}")

    # Optional torch.compile (PyTorch 2.0+ automatic Triton kernel fusion)
    if args.compile:
        try:
            if is_master:
                print("🚀 Compiling model with torch.compile(mode='reduce-overhead')...")
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            if is_master:
                print(f"⚠️ torch.compile not supported or failed: {e}")

    # Wrap model with DDP
    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = GaLoreAdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        rank=args.galore_rank,
        update_proj_gap=50
    )

    dataset = SyntheticStoryDataset(num_samples=1000, seq_len=64, vocab_size=args.vocab_size)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if is_distributed else None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        pin_memory=(device.startswith("cuda"))
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * (len(dataloader) // args.grad_accum_steps + 1)
    )

    # AMP setup
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    device_type = "cuda" if "cuda" in device else "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device_type == "cuda" and amp_dtype == torch.float16))

    for epoch in range(1, args.epochs + 1):
        if is_distributed and sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch:02d}/{args.epochs:02d}]") if is_master else dataloader

        for step, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device_type, dtype=amp_dtype, enabled=args.amp):
                logits = model(inputs)
                loss = F.cross_entropy(logits.view(-1, args.vocab_size), targets.view(-1))
                loss = loss / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(dataloader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()

            raw_loss = loss.item() * args.grad_accum_steps
            total_loss += raw_loss
            if is_master:
                ppl = math.exp(min(raw_loss, 100))
                pbar.set_postfix({"loss": f"{raw_loss:.4f}", "ppl": f"{ppl:.2f}"})

    if is_distributed:
        torch.distributed.destroy_process_group()

    # Master node exports the binary
    if is_master:
        raw_model = model.module if is_distributed else model
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        export_binary(raw_model, args.out_bin)


if __name__ == "__main__":
    main()
