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


try:
    from causal_conv1d import causal_conv1d_fn
    HAS_CAUSAL_CONV1D = True
except ImportError:
    HAS_CAUSAL_CONV1D = False

try:
    from triton_kernels import (
        fused_delta_ssm_scan,
        fused_cross_entropy,
        FusedRMSNorm,
        fused_silu_gating,
    )
except ImportError:
    try:
        from .triton_kernels import (
            fused_delta_ssm_scan,
            fused_cross_entropy,
            FusedRMSNorm,
            fused_silu_gating,
        )
    except ImportError:
        fused_delta_ssm_scan = None
        fused_cross_entropy = None
        FusedRMSNorm = RMSNorm
        fused_silu_gating = None


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

        # Fast-path: use Tri Dao's fused CUDA causal_conv1d_fn if available
        if HAS_CAUSAL_CONV1D and u.is_cuda:
            u_conv = causal_conv1d_fn(
                u.transpose(1, 2).contiguous(),
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation="silu"
            ).transpose(1, 2)
        else:
            u_conv = self.conv1d(u.transpose(1, 2))[:, :, :L].transpose(1, 2)
            u_conv = F.silu(u_conv)

        B_t = self.b_proj(u_conv)
        C_t = self.c_proj(u_conv)
        decay = torch.sigmoid(self.decay_fast)
        u_scalar = u_conv.mean(dim=-1, keepdim=True)
        x_in = B_t * u_scalar

        if cached_state is None:
            if fused_delta_ssm_scan is not None and x.is_cuda:
                y_t, next_state = fused_delta_ssm_scan(x_in, decay, C_t)
            else:
                # Vectorized Parallel Causal Scan fallback
                t_idx = torch.arange(L, device=x.device)
                diff = (t_idx[:, None] - t_idx[None, :]).clamp(min=0)
                mask = (t_idx[:, None] >= t_idx[None, :]).float()
                decay_mat = (decay.view(1, 1, -1) ** diff.unsqueeze(-1)) * mask.unsqueeze(-1)
                h_seq = torch.einsum('l k s, b k s -> b l s', decay_mat, x_in)
                y_t = (C_t * h_seq).sum(dim=-1, keepdim=True)
                next_state = h_seq[:, -1, :]
        else:
            # Step-by-step state update for token-by-token cached inference
            h = cached_state
            y_list = []
            for t in range(L):
                h = decay * h + x_in[:, t, :]
                y_list.append((C_t[:, t, :] * h).sum(dim=-1, keepdim=True))
            y_t = torch.stack(y_list, dim=1)
            next_state = h

        y_ssm = y_t.expand(-1, -1, D)
        y = (u_conv + y_ssm) * F.silu(gate)
        return self.out_proj(y), next_state


class BitMCSSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 32, tau: float = 0.85):
        super().__init__()
        self.norm1 = FusedRMSNorm(d_model)
        self.ssm = DeltaSSMBlock(d_model=d_model, d_state=d_state, tau=tau)
        self.norm2 = FusedRMSNorm(d_model)
        self.ffn_in = HBitLinear(d_model, d_model * 4, tau=tau, use_hadamard=False)
        self.ffn_out = HBitLinear(d_model * 2, d_model, tau=tau, use_hadamard=True)

    def forward(self, x: torch.Tensor, cached_state=None):
        ssm_out, next_state = self.ssm(self.norm1(x), cached_state)
        x = x + ssm_out
        ffn_p = self.ffn_in(self.norm2(x))
        f1, f2 = ffn_p.chunk(2, dim=-1)
        if fused_silu_gating is not None and x.is_cuda:
            gated = fused_silu_gating(f1, f2)
        else:
            gated = F.silu(f1) * f2
        x = x + self.ffn_out(gated)
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
        self.final_norm = FusedRMSNorm(d_model)
        self.lm_head = HBitLinear(d_model, vocab_size, tau=tau, use_hadamard=False)

    def extract_features(self, idx: torch.Tensor):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x, _ = block(x)
        return self.final_norm(x)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, chunk_size: int = 64):
        hidden = self.extract_features(idx)
        if targets is None:
            return self.lm_head(hidden)

        B, L, D = hidden.shape
        flat_hidden = hidden.view(-1, D)
        flat_targets = targets.view(-1)
        total_tokens = flat_targets.numel()

        # Fused Flash Cross-Entropy (Triton)
        if fused_cross_entropy is not None and flat_hidden.is_cuda:
            # When Triton is available, compute Flash Cross-Entropy directly
            logits = self.lm_head(flat_hidden)
            return fused_cross_entropy(logits, flat_targets)

        if chunk_size <= 0 or total_tokens <= chunk_size:
            logits = self.lm_head(flat_hidden)
            return F.cross_entropy(logits, flat_targets)

        # Chunked Cross-Entropy fallback for PyTorch without Triton:
        total_loss = 0.0
        for i in range(0, total_tokens, chunk_size):
            h_chunk = flat_hidden[i : i + chunk_size]
            t_chunk = flat_targets[i : i + chunk_size]
            logits_chunk = self.lm_head(h_chunk)
            chunk_loss = F.cross_entropy(logits_chunk, t_chunk, reduction="sum")
            total_loss = total_loss + chunk_loss

        return total_loss / total_tokens


class SyntheticStoryDataset(Dataset):
    def __init__(self, num_samples: int = 1000, seq_len: int = 64, vocab_size: int = 50257):
        self.data = []
        patterns = [
            [7454, 2402, 257, 640, 11, 20037, 2497, 257, 44152, 1757, 393, 20037, 547, 257, 2833, 4037],
            [464, 1310, 3290, 3058, 284, 862, 287, 262, 3867, 13, 6342, 547, 1266, 284, 4015],
            [15496, 11, 262, 3867, 373, 257, 1255, 379, 262, 983, 13, 1119, 547, 1363, 284, 883],
            [1856, 4060, 11, 670, 318, 257, 1266, 1778, 13, 383, 1540, 423, 262, 1856, 3556, 326],
            [2061, 318, 257, 3745, 1104, 3838, 13, 383, 2664, 284, 467, 319, 262, 3838, 290, 4831],
            [8241, 338, 262, 1693, 11, 284, 2402, 257, 640, 13, 679, 12056, 262, 3804, 11, 290, 547],
            [2065, 318, 257, 2220, 2438, 13, 383, 3000, 284, 1207, 262, 3948, 11, 290, 1239, 373],
            [198, 40, 1101, 262, 3867, 13, 383, 550, 257, 649, 11, 290, 1119, 547, 1363, 13]
        ]
        for _ in range(num_samples):
            seq = []
            while len(seq) < seq_len + 1:
                seq.extend(patterns[torch.randint(0, len(patterns), (1,)).item()])
            seq = seq[:seq_len + 1]
            self.data.append(torch.tensor(seq, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]


def export_binary(model: BitMCSSM, out_path: str):
    """
    Exports BitMCSSM to native packed 2-bit binary format for C++ Zero-GEMM inference.
    """
    print(f"📦 Exporting model to 2-bit binary: {out_path}...")
    with open(out_path, "wb") as f:
        # Magic 'BSSM' = 0x4D535342
        header = struct.pack(
            "<IIIII",
            0x4D535342,
            model.vocab_size,
            model.d_model,
            model.n_layers,
            model.d_state
        )
        f.write(header)

        # 1. Token Embeddings (FP32)
        emb_weight = model.tok_emb.weight.detach().cpu().to(torch.float32).numpy()
        f.write(emb_weight.tobytes())

        # 2. Sequential SSM / Transformer Blocks
        for i, block in enumerate(model.blocks):
            # RMSNorm 1
            f.write(block.norm1.weight.detach().cpu().to(torch.float32).numpy().tobytes())

            # SSM In-Proj (Packed 2-bit + scale)
            g_in, p_in = pack_ternary_weights(block.ssm.in_proj.weight)
            f.write(struct.pack("<f", g_in) + p_in)

            # Conv1D Weight (FP32) & Bias (FP32)
            f.write(block.ssm.conv1d.weight.detach().cpu().to(torch.float32).numpy().tobytes())
            if block.ssm.conv1d.bias is not None:
                f.write(block.ssm.conv1d.bias.detach().cpu().to(torch.float32).numpy().tobytes())
            else:
                f.write(np.zeros(model.d_model, dtype=np.float32).tobytes())

            # B Proj & C Proj
            g_b, p_b = pack_ternary_weights(block.ssm.b_proj.weight)
            f.write(struct.pack("<f", g_b) + p_b)

            g_c, p_c = pack_ternary_weights(block.ssm.c_proj.weight)
            f.write(struct.pack("<f", g_c) + p_c)

            # Fast Decay (FP32)
            decay_f = torch.sigmoid(block.ssm.decay_fast).detach().cpu().to(torch.float32).numpy()
            f.write(decay_f.tobytes())

            # SSM Out Proj
            g_out, p_out = pack_ternary_weights(block.ssm.out_proj.weight)
            f.write(struct.pack("<f", g_out) + p_out)

            # RMSNorm 2
            f.write(block.norm2.weight.detach().cpu().to(torch.float32).numpy().tobytes())

            # FFN In Proj & Out Proj
            g_fin, p_fin = pack_ternary_weights(block.ffn_in.weight)
            f.write(struct.pack("<f", g_fin) + p_fin)

            g_fout, p_fout = pack_ternary_weights(block.ffn_out.weight)
            f.write(struct.pack("<f", g_fout) + p_fout)

        # 3. Final RMSNorm
        f.write(model.final_norm.weight.detach().cpu().to(torch.float32).numpy().tobytes())

        # 4. LM Head (Packed 2-bit + scale)
        g_head, p_head = pack_ternary_weights(model.lm_head.weight)
        f.write(struct.pack("<f", g_head) + p_head)

    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"✅ Exported successfully! Binary size: {file_size_mb:.2f} MB\n")


def main():
    parser = argparse.ArgumentParser(description="BitMC-SSM Multi-GPU Pre-training & 2-bit Export Engine")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Per-device batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1.0e-3, help="Peak learning rate")
    parser.add_argument("--galore_rank", type=int, default=16, help="GaLore low-rank projection rank")
    parser.add_argument("--d_model", type=int, default=384, help="Model hidden dimension")
    parser.add_argument("--n_layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--d_state", type=int, default=32, help="SSM state dimension")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision (FP16/BF16)")
    parser.add_argument("--compile", action="store_true", help="Enable PyTorch 2.0+ torch.compile()")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "smollm", "tinystories"], help="Dataset")
    parser.add_argument("--dataset_subset", type=str, default="cosmopedia-v2", help="SmolLM subset (cosmopedia-v2, stories, etc.)")
    parser.add_argument("--data_bin", type=str, default=None, help="Path to pre-tokenized memory-mapped .bin file (e.g. data/tinystories.bin)")
    parser.add_argument("--num_samples", type=int, default=25000, help="Number of samples/sequences")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument("--chunk_size", type=int, default=64, help="Chunk size for fused chunked cross-entropy loss")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader background worker processes")
    parser.add_argument("--save_ckpt_dir", type=str, default=None, help="Directory to save PyTorch checkpoints (.pt)")
    parser.add_argument("--save_every_epochs", type=int, default=1, help="Epoch interval for saving checkpoints")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint .pt file to resume from")
    parser.add_argument("--out_bin", type=str, default="model.bin", help="Output 2-bit binary path")
    args = parser.parse_args()

    # Multi-GPU / Distributed Setup (torchrun / DDP)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1

    if is_distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
            device = f"cuda:{local_rank}"
        else:
            backend = "gloo"
            device = "cpu"

        torch.distributed.init_process_group(backend=backend)
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        rank = 0
        world_size = 1
        device = args.device

    is_master = (rank == 0)

    if is_master:
        has_cuda = torch.cuda.is_available() and device != "cpu"
        triton_active = (fused_cross_entropy is not None and has_cuda)

        print("======================================================================")
        print(f"⚡ Bit-MC-SSM Training on {world_size}x Device(s) (DDP={is_distributed}, backend={backend if is_distributed else 'none'})")
        print(f"   Config: d_model={args.d_model}, layers={args.n_layers}, d_state={args.d_state}")
        print(f"   Optimizations: AMP={args.amp}, compile={args.compile}, grad_accum={args.grad_accum_steps}, chunk_loss={args.chunk_size}")
        print(f"   Dataset: {args.dataset} ({args.dataset_subset if args.dataset == 'smollm' else 'synthetic'}), Samples={args.num_samples:,}")
        print(f"   Effective Batch Size: {args.batch_size * args.grad_accum_steps * world_size}")
        print("----------------------------------------------------------------------")
        print("🚀 Kernel Acceleration Engine Status:")
        print(f"   • Fused Flash Cross-Entropy : {'[⚡ ACTIVE (Triton SRAM)]' if triton_active else '[🛡️ FALLBACK (PyTorch)]'}")
        print(f"   • Fused Delta-SSM Scan     : {'[⚡ ACTIVE (Triton Scan)]' if (fused_delta_ssm_scan is not None and has_cuda) else '[🛡️ FALLBACK (PyTorch)]'}")
        print(f"   • Fused RMSNorm & SiLU     : {'[⚡ ACTIVE (Triton Fused)]' if triton_active else '[🛡️ FALLBACK (PyTorch)]'}")
        print(f"   • CausalConv1D Engine      : {'[⚡ ACTIVE (CUDA Kernel)]' if HAS_CAUSAL_CONV1D else '[🛡️ FALLBACK (PyTorch Depthwise)]'}")
        print("----------------------------------------------------------------------")
        if args.save_ckpt_dir:
            print(f"   Checkpoint Saving: every {args.save_every_epochs} epoch(s) to '{args.save_ckpt_dir}'")
        if args.resume_from:
            print(f"   Resuming from: '{args.resume_from}'")
        print("======================================================================")

    model = BitMCSSM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state
    ).to(device)

    start_epoch = 1
    best_loss = float("inf")

    # Resume from checkpoint if specified
    if args.resume_from and os.path.exists(args.resume_from):
        if is_master:
            print(f"🔄 Loading checkpoint from {args.resume_from}...")
        ckpt = torch.load(args.resume_from, map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_loss = ckpt.get("loss", float("inf"))
            if is_master:
                print(f"✅ Checkpoint loaded! Resuming from Epoch {start_epoch}")
        else:
            model.load_state_dict(ckpt)
            if is_master:
                print("✅ Model weights loaded successfully!")

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
        if torch.cuda.is_available():
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
                bucket_cap_mb=25
            )
        else:
            model = DDP(model)

    optimizer = GaLoreAdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        rank=args.galore_rank,
        update_proj_gap=50
    )

    if args.data_bin and os.path.exists(args.data_bin):
        class PretokenizedMemmapDataset(Dataset):
            def __init__(self, bin_path: str, seq_len: int = 128, max_samples: int = -1):
                self.seq_len = seq_len
                self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
                self.total_tokens = len(self.data)
                total_chunks = (self.total_tokens - 1) // seq_len
                if max_samples > 0:
                    self.num_samples = min(total_chunks, max_samples)
                else:
                    self.num_samples = total_chunks

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                start = idx * self.seq_len
                chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
                x = torch.from_numpy(chunk[:-1])
                y = torch.from_numpy(chunk[1:])
                return x, y

        if is_master:
            print(f"⚡ Loading pre-tokenized memory-mapped binary: {args.data_bin}")
        dataset = PretokenizedMemmapDataset(args.data_bin, seq_len=args.seq_len, max_samples=args.num_samples)
        if is_master:
            print(f"📊 Total Available Samples: {len(dataset):,} ({dataset.total_tokens:,} tokens)")
    elif args.dataset in ["smollm", "tinystories"]:
        from datasets import load_dataset
        from transformers import GPT2TokenizerFast

        if is_master:
            target_ds = "roneneldan/TinyStories" if args.dataset == "tinystories" else f"HuggingFaceTB/smollm-corpus ({args.dataset_subset})"
            print(f"📥 Loading dataset: {target_ds}...")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        if args.dataset == "tinystories":
            raw_data = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        else:
            raw_data = load_dataset("HuggingFaceTB/smollm-corpus", args.dataset_subset, split="train", streaming=True)

        samples = []
        count = 0
        for item in raw_data:
            text = item.get("text", "").strip()
            if len(text) < 50:
                continue
            toks = tokenizer.encode(text)
            for start_idx in range(0, len(toks) - args.seq_len, args.seq_len):
                chunk = toks[start_idx : start_idx + args.seq_len + 1]
                if len(chunk) == args.seq_len + 1:
                    samples.append(torch.tensor(chunk, dtype=torch.long))
                    count += 1
                    if count >= args.num_samples:
                        break
            if count >= args.num_samples:
                break

        class ListDataset(Dataset):
            def __init__(self, s):
                self.s = s
            def __len__(self):
                return len(self.s)
            def __getitem__(self, idx):
                seq = self.s[idx]
                return seq[:-1], seq[1:]

        dataset = ListDataset(samples)
    else:
        dataset = SyntheticStoryDataset(num_samples=args.num_samples, seq_len=args.seq_len, vocab_size=args.vocab_size)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if is_distributed else None

    use_workers = args.num_workers if (torch.cuda.is_available() and args.num_workers > 0) else 0
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=use_workers,
        pin_memory=(device.startswith("cuda")),
        persistent_workers=(use_workers > 0),
        prefetch_factor=2 if use_workers > 0 else None
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * (len(dataloader) // args.grad_accum_steps + 1)
    )

    # AMP setup
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    device_type = "cuda" if "cuda" in device else "cpu"
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device_type == "cuda" and amp_dtype == torch.float16))

    if args.resume_from and os.path.exists(args.resume_from) and "scaler_state_dict" in ckpt:
        if scaler.is_enabled() and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])

    for epoch in range(start_epoch, args.epochs + 1):
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
                # Chunked cross-entropy prevents huge 400MB+ logits allocations
                loss = model(inputs, targets=targets, chunk_size=args.chunk_size)
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

        # Epoch checkpointing (Master Rank)
        if is_master and args.save_ckpt_dir:
            os.makedirs(args.save_ckpt_dir, exist_ok=True)
            avg_epoch_loss = total_loss / max(1, len(dataloader))
            raw_model = model.module if is_distributed else model
            raw_model = getattr(raw_model, "_orig_mod", raw_model)

            ckpt_payload = {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                "config": vars(args),
                "loss": avg_epoch_loss
            }

            # Save latest checkpoint
            last_path = os.path.join(args.save_ckpt_dir, "ckpt_last.pt")
            torch.save(ckpt_payload, last_path)

            # Save best checkpoint
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_path = os.path.join(args.save_ckpt_dir, "ckpt_best.pt")
                torch.save(ckpt_payload, best_path)

            # Save periodic checkpoint
            if epoch % args.save_every_epochs == 0:
                epoch_path = os.path.join(args.save_ckpt_dir, f"ckpt_epoch_{epoch:02d}.pt")
                torch.save(ckpt_payload, epoch_path)
                print(f"💾 Checkpoint saved: {epoch_path} (Loss: {avg_epoch_loss:.4f})")

    if is_distributed:
        torch.distributed.destroy_process_group()

    # Master node exports the binary
    if is_master:
        raw_model = model.module if is_distributed else model
        raw_model = getattr(raw_model, "_orig_mod", raw_model)
        export_binary(raw_model, args.out_bin)


if __name__ == "__main__":
    main()
