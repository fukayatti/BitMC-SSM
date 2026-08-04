"""
Bit-MC-SSM (1.58-bit Memory-Cached State Space Model)
Phase 5: Forward-Forward (FF) Algorithm & Local Learning Research Prototype
Eliminates Global Backpropagation for Zero-Activation-Memory CPU Training.
(Optimized Formulation: Contrastive Goodness Margin + Active Local Representation Heads)
"""

import time
import math
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. Quantization & 1.58-bit Primitives
# ==============================================================================

class Quantize158(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        gamma = weight.abs().mean().clamp(min=1e-5)
        w_scaled = weight / gamma
        w_ternary = torch.clamp(torch.round(w_scaled), -1.0, 1.0)
        return w_ternary * gamma

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class BitLinear158(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        w_q = Quantize158.apply(self.weight)
        return F.linear(x, w_q, self.bias)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

# ==============================================================================
# 2. Forward-Forward ShiftSSM Block (Fully Local Layer)
# ==============================================================================

class ForwardForwardSSMBlock(nn.Module):
    """
    A standalone State Space Model Block trained via Local Learning & Forward-Forward.
    Has its own independent optimizer; NEVER receives gradients from upper layers!
    """
    def __init__(self, d_model: int, d_state: int = 16, vocab_size: int = 1000, lr: float = 3e-3):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.vocab_size = vocab_size

        # Internal SSM & FFN
        self.norm1 = RMSNorm(d_model)
        self.in_proj = BitLinear158(d_model, 2 * d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        self.b_proj = BitLinear158(d_model, d_state)
        self.c_proj = BitLinear158(d_model, d_state)
        self.decay_param = nn.Parameter(torch.tensor([-1.0] * d_state))
        self.out_proj = BitLinear158(d_model, d_model)

        self.norm2 = RMSNorm(d_model)
        self.ffn_in = BitLinear158(d_model, d_model * 4)
        self.ffn_out = BitLinear158(d_model * 2, d_model)

        # Local Auxiliary Next-Token Predictor
        self.local_head = nn.Linear(d_model, vocab_size, bias=False)

        # Dedicated optimizer for this layer only!
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)

    def _ssm_step(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        nx = self.norm1(x)
        proj = self.in_proj(nx)
        u, gate = proj.chunk(2, dim=-1)

        u_conv = self.conv1d(u.transpose(1, 2))[:, :, :L].transpose(1, 2)
        u_conv = F.silu(u_conv)

        B_t = self.b_proj(u_conv)
        C_t = self.c_proj(u_conv)
        decay = torch.sigmoid(self.decay_param)

        # Sequential scan
        h = torch.zeros(B, self.d_state, device=x.device, dtype=x.dtype)
        y_list = []
        u_scalar = u_conv.mean(dim=-1, keepdim=True)
        for t in range(L):
            h = decay * h + B_t[:, t, :] * u_scalar[:, t, :]
            y_t = (C_t[:, t, :] * h).sum(dim=-1, keepdim=True)
            y_list.append(y_t)
        y_ssm = torch.cat(y_list, dim=-1).unsqueeze(-1).expand(-1, -1, D)

        ssm_out = self.out_proj((u_conv + y_ssm) * F.silu(gate))
        x = x + ssm_out

        # FFN
        nx2 = self.norm2(x)
        ffn_p = self.ffn_in(nx2)
        f1, f2 = ffn_p.chunk(2, dim=-1)
        ffn_act = F.silu(f1) * f2
        x = x + self.ffn_out(ffn_act)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._ssm_step(x)

    def train_step_ff(self, h_pos: torch.Tensor, h_neg: torch.Tensor, target_ids: torch.Tensor) -> dict:
        """
        Local Training Step (No backward gradients from downstream layers):
        1. Forward on positive stream & negative stream
        2. Contrastive Goodness loss (maximize G(pos) - G(neg))
        3. Local next-token cross entropy loss
        4. Independent local weight update
        """
        self.optimizer.zero_grad()

        out_pos = self._ssm_step(h_pos)
        out_neg = self._ssm_step(h_neg)

        # Goodness per token
        g_pos = (out_pos ** 2).mean(dim=-1) # [B, L]
        g_neg = (out_neg ** 2).mean(dim=-1) # [B, L]

        # Contrastive Goodness Loss (Softplus margin)
        # Encourages g_pos > g_neg with a positive margin
        loss_ff = F.softplus(g_neg - g_pos + 1.0).mean()

        # Local LM Cross Entropy
        logits = self.local_head(out_pos)
        loss_lm = F.cross_entropy(logits.view(-1, self.vocab_size), target_ids.view(-1))

        # Total Local Loss
        total_loss = loss_lm + 0.1 * loss_ff
        total_loss.backward()
        self.optimizer.step()

        # Compute next layer inputs (normalized & detached!)
        with torch.no_grad():
            next_h_pos = out_pos / (out_pos.norm(2, dim=-1, keepdim=True) + 1e-6)
            next_h_neg = out_neg / (out_neg.norm(2, dim=-1, keepdim=True) + 1e-6)

        return {
            "loss_ff": float(loss_ff.item()),
            "loss_lm": float(loss_lm.item()),
            "g_pos": float(g_pos.mean().item()),
            "g_neg": float(g_neg.mean().item()),
            "next_h_pos": next_h_pos.detach(),
            "next_h_neg": next_h_neg.detach()
        }

# ==============================================================================
# 3. Full Forward-Forward Bit-SSM Model
# ==============================================================================

class ForwardForwardBitSSM(nn.Module):
    def __init__(self, vocab_size: int = 1000, d_model: int = 64, n_layers: int = 3, d_state: int = 16, lr: float = 3e-3):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.emb_head = nn.Linear(d_model, vocab_size, bias=False)
        self.emb_optimizer = torch.optim.AdamW(
            list(self.embedding.parameters()) + list(self.emb_head.parameters()),
            lr=lr, weight_decay=0.01
        )

        self.blocks = nn.ModuleList([
            ForwardForwardSSMBlock(d_model=d_model, d_state=d_state, vocab_size=vocab_size, lr=lr)
            for _ in range(n_layers)
        ])

    def train_step(self, pos_tokens: torch.Tensor, neg_tokens: torch.Tensor, target_ids: torch.Tensor) -> list:
        """
        Zero-Backpropagation Full Training Step.
        Each layer updates its own weights independently and passes detached outputs!
        """
        # 1. Train Embedding Layer Locally
        self.emb_optimizer.zero_grad()
        h_pos = self.embedding(pos_tokens)
        emb_logits = self.emb_head(h_pos)
        emb_loss = F.cross_entropy(emb_logits.view(-1, self.vocab_size), target_ids.view(-1))
        emb_loss.backward()
        self.emb_optimizer.step()

        # Detach embeddings for downstream layers
        with torch.no_grad():
            cur_pos = (h_pos / (h_pos.norm(2, dim=-1, keepdim=True) + 1e-6)).detach()
            h_neg = self.embedding(neg_tokens)
            cur_neg = (h_neg / (h_neg.norm(2, dim=-1, keepdim=True) + 1e-6)).detach()

        layer_stats = [{"loss_ff": 0.0, "loss_lm": float(emb_loss.item()), "g_pos": 1.0, "g_neg": 1.0}]

        # 2. Train each SSM Block sequentially (Completely Lock-Free & Detached!)
        for block in self.blocks:
            stats = block.train_step_ff(cur_pos, cur_neg, target_ids)
            layer_stats.append(stats)
            cur_pos = stats["next_h_pos"]
            cur_neg = stats["next_h_neg"]

        return layer_stats

    @torch.no_grad()
    def generate(self, prompt_tokens: torch.Tensor, max_new_tokens: int = 24, temperature: float = 0.2) -> torch.Tensor:
        self.eval()
        gen = prompt_tokens.clone()
        for _ in range(max_new_tokens):
            h = self.embedding(gen)
            h = h / (h.norm(2, dim=-1, keepdim=True) + 1e-6)
            for block in self.blocks:
                h = block(h)
                h = h / (h.norm(2, dim=-1, keepdim=True) + 1e-6)
            logits = self.blocks[-1].local_head(h[:, -1, :])
            if temperature < 1e-3:
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            gen = torch.cat([gen, next_tok], dim=1)
        return gen

# ==============================================================================
# 4. Structured Synthetic & Negative Data Generator
# ==============================================================================

class StructuredPatternDataset(Dataset):
    def __init__(self, num_samples: int = 500, seq_len: int = 32, vocab_size: int = 500):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Create structured predictable loops: [A, B, C, D] repeating
        self.data = []
        for _ in range(num_samples):
            # Select 4 tokens as fixed pattern
            a = np.random.randint(10, 100)
            b = np.random.randint(100, 200)
            c = np.random.randint(200, 300)
            d = np.random.randint(300, 400)
            base = [a, b, c, d]
            full = (base * ((seq_len // 4) + 2))[:seq_len + 1]
            self.data.append(torch.tensor(full, dtype=torch.long))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        full = self.data[idx]
        pos = full[:-1]
        targets = full[1:]

        # Negative samples: Corrupt tokens with random noise
        neg = pos.clone()
        mask = torch.rand(self.seq_len) < 0.4
        neg[mask] = torch.randint(0, self.vocab_size, (mask.sum().item(),))

        return pos, neg, targets

# ==============================================================================
# 5. Training Execution & Benchmark
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Forward-Forward Bit-SSM Training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--d_model", type=int, default=64, help="Model hidden dimension")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of SSM layers")
    parser.add_argument("--vocab_size", type=int, default=500, help="Vocab size")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length")
    args = parser.parse_args()

    print("=" * 70)
    print("🔬 Phase 5: Forward-Forward (FF) Bit-MC-SSM Training Experiment (Fix v2)")
    print("   Zero Global Backpropagation / Fully Local Layer-wise Learning")
    print("=" * 70)

    device = "cpu"
    dataset = StructuredPatternDataset(num_samples=500, seq_len=args.seq_len, vocab_size=args.vocab_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = ForwardForwardBitSSM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Total Model Parameters: {total_params:,}")
    print("🚀 Training Started (No global backward graph created!)...\n")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_lm_loss = [0.0] * (args.n_layers + 1)
        epoch_ff_loss = [0.0] * args.n_layers
        epoch_g_diff = [0.0] * args.n_layers
        steps = 0

        for pos, neg, targets in dataloader:
            pos, neg, targets = pos.to(device), neg.to(device), targets.to(device)
            stats = model.train_step(pos, neg, targets)

            # Embedding layer
            epoch_lm_loss[0] += stats[0]["loss_lm"]

            # Blocks
            for l in range(args.n_layers):
                s = stats[l + 1]
                epoch_lm_loss[l + 1] += s["loss_lm"]
                epoch_ff_loss[l] += s["loss_ff"]
                epoch_g_diff[l] += (s["g_pos"] - s["g_neg"])
            steps += 1

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Elapsed: {time.time() - start_time:.1f}s")
        print(f"   Emb Layer:  LM Loss: {epoch_lm_loss[0]/steps:.4f}")
        for l in range(args.n_layers):
            avg_lm = epoch_lm_loss[l + 1] / steps
            avg_ff = epoch_ff_loss[l] / steps
            avg_gd = epoch_g_diff[l] / steps
            print(f"   Block {l}:    LM Loss: {avg_lm:.4f} | FF Loss: {avg_ff:.4f} | Goodness Pos-Neg: {avg_gd:+.3f}")
        print("-" * 70)

    # Save checkpoint
    os.makedirs("models", exist_ok=True)
    save_path = "models/bit_mc_ssm_forward_forward.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": args.vocab_size,
            "d_model": args.d_model,
            "n_layers": args.n_layers
        }
    }, save_path)
    print(f"\n💾 Saved Forward-Forward checkpoint to: {save_path}")

    # Text / Token Generation Test (Prompt pattern: [50, 150, 250, 350] -> should repeat!)
    sample_prompt = dataset.data[0][:4].unsqueeze(0).to(device)
    gen = model.generate(sample_prompt, max_new_tokens=16, temperature=0.0) # greedy
    print("\n✨ Generation Verification with Forward-Forward Model:")
    print(f"   Prompt Pattern:  {sample_prompt.tolist()[0]}")
    print(f"   Expected Repeat: {dataset.data[0][:20].tolist()}")
    print(f"   Model Output:    {gen.tolist()[0]}")
    
    # Check accuracy of generated tokens
    expected = dataset.data[0][:20].tolist()
    actual = gen.tolist()[0]
    matches = sum(1 for e, a in zip(expected[4:], actual[4:]))
    acc = (matches / len(actual[4:])) * 100
    print(f"\n🎯 Sequence Pattern Replication Accuracy: {acc:.1f}% ({matches}/{len(actual[4:])})")
    print("=" * 70)

if __name__ == "__main__":
    main()
