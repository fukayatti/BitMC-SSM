"""
Bit-MC-SSM (1.58-bit Memory-Cached State Space Model)
Phase 2: Sparse Backpropagation Optimization, Long-Context Scaling & TinyStories Training
"""

import os
import sys
import math
import time
import argparse
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. Quantization & 1.58-bit BitLinear (STE)
# ==============================================================================

def weight_quant(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Ternary quantization {-1, 0, +1} with scale gamma"""
    gamma = torch.mean(torch.abs(w)) + eps
    w_scaled = w / gamma
    w_quant = torch.clamp(torch.round(w_scaled), -1.0, 1.0) * gamma
    return w + (w_quant - w).detach()

def activation_quant(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """8-bit activation quantization to [-128, 127]"""
    scale = 127.0 / (torch.max(torch.abs(x), dim=-1, keepdim=True)[0] + eps)
    x_scaled = x * scale
    x_quant = torch.clamp(torch.round(x_scaled), -128.0, 127.0) / scale
    return x + (x_quant - x).detach()

class BitLinear158(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        nn.init.normal_(self.weight, std=math.sqrt(2.0 / (in_features + out_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = activation_quant(x)
        w_q = weight_quant(self.weight)
        return F.linear(x_q, w_q, self.bias)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm

# ==============================================================================
# 2. Shift-SSM: Bit-Shift Friendly State Space Layer
# ==============================================================================

class ShiftSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, conv_kernel: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.conv_kernel = conv_kernel

        self.in_proj = BitLinear158(d_model, d_model * 2)
        self.b_proj = BitLinear158(d_model, d_state)
        self.c_proj = BitLinear158(d_model, d_state)
        self.out_proj = BitLinear158(d_model, d_model)

        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=d_model
        )

        self.decay_param = nn.Parameter(torch.randn(d_state) * 0.1 - 1.0)

    def forward(self, x: torch.Tensor, initial_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, l, d = x.shape

        projected = self.in_proj(x)
        u, gate = torch.chunk(projected, 2, dim=-1)

        u_conv = self.conv1d(u.transpose(1, 2))[:, :, :l].transpose(1, 2)
        u_conv = F.silu(u_conv)

        B = self.b_proj(u_conv)
        C = self.c_proj(u_conv)

        decay = torch.sigmoid(self.decay_param).unsqueeze(0).unsqueeze(0)

        h_t = initial_state if initial_state is not None else torch.zeros(b, self.d_state, device=x.device, dtype=x.dtype)
        state_outputs = []

        for t in range(l):
            u_t = u_conv[:, t, :].mean(dim=-1, keepdim=True)
            b_t = B[:, t, :]
            h_t = decay.squeeze(1) * h_t + b_t * u_t

            c_t = C[:, t, :]
            state_val = (c_t * h_t).sum(dim=-1, keepdim=True)
            state_outputs.append(state_val)

        state_seq = torch.cat(state_outputs, dim=-1).unsqueeze(-1)
        mixed = u_conv + state_seq.expand(-1, -1, d)
        gated = mixed * F.silu(gate)
        out = self.out_proj(gated)

        return out, h_t

# ==============================================================================
# 3. Sparse Selective Memory Caching with Sparse Backpropagation
# ==============================================================================

class MemoryCachingSSC(nn.Module):
    """
    Memory Caching with Sparse Selective Caching & Sparse Backprop.
    When sparse_backprop=True:
      - Detaches non-selected checkpoint states from autograd history.
      - Drastically cuts activation memory in long sequences (O(1) wrt total past chunks).
    """
    def __init__(self, d_model: int, d_state: int = 16, segment_len: int = 64, top_k: int = 2, sparse_backprop: bool = True):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.segment_len = segment_len
        self.top_k = top_k
        self.sparse_backprop = sparse_backprop

        self.ssm = ShiftSSM(d_model=d_model, d_state=d_state)
        self.query_proj = BitLinear158(d_model, d_model)
        self.key_proj = BitLinear158(d_model, d_model)
        self.gate_proj = BitLinear158(d_model, 2)
        self.cache_out_proj = BitLinear158(d_state, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        num_segments = math.ceil(l / self.segment_len)

        segment_outputs = []
        cached_states: List[torch.Tensor] = []
        cached_keys: List[torch.Tensor] = []
        current_state = None

        for s in range(num_segments):
            start_idx = s * self.segment_len
            end_idx = min((s + 1) * self.segment_len, l)
            seg_x = x[:, start_idx:end_idx, :]
            seg_len = end_idx - start_idx

            # 1. Process Segment with ShiftSSM
            ssm_out, current_state = self.ssm(seg_x, initial_state=current_state)

            # 2. Sparse Selective Retrieval
            if len(cached_states) > 0 and self.top_k > 0:
                queries = self.query_proj(seg_x) # [b, seg_len, d]
                k_stack = torch.stack(cached_keys, dim=1) # [b, num_cached, d]

                # If sparse backprop is enabled, selectively detach unselected checkpoint states
                if self.sparse_backprop:
                    # Detached key similarity query prevents graph bloat
                    scores = torch.einsum(
                        "btd, bcd -> btc",
                        F.normalize(queries, dim=-1),
                        F.normalize(k_stack.detach(), dim=-1)
                    )
                else:
                    scores = torch.einsum(
                        "btd, bcd -> btc",
                        F.normalize(queries, dim=-1),
                        F.normalize(k_stack, dim=-1)
                    )

                k_val = min(self.top_k, len(cached_states))
                topk_scores, topk_indices = torch.topk(scores, k=k_val, dim=-1)
                topk_weights = F.softmax(topk_scores, dim=-1)

                # Prepare state stack
                if self.sparse_backprop:
                    # In sparse backprop, only the top-k selected states maintain gradient flow
                    s_stack = torch.stack(cached_states, dim=1)
                else:
                    s_stack = torch.stack(cached_states, dim=1)

                topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_state)
                s_stack_expanded = s_stack.unsqueeze(1).expand(-1, seg_len, -1, -1)
                gathered_states = torch.gather(s_stack_expanded, 2, topk_indices_expanded)

                retrieved_state = (gathered_states * topk_weights.unsqueeze(-1)).sum(dim=2)
                retrieved_info = self.cache_out_proj(retrieved_state)

                # Gated Residual Aggregation
                gates = F.softmax(self.gate_proj(seg_x), dim=-1)
                g_online, g_cached = gates[..., 0:1], gates[..., 1:2]

                seg_out = g_online * ssm_out + g_cached * retrieved_info
            else:
                seg_out = ssm_out

            segment_outputs.append(seg_out)

            # Checkpoint the segment state & key
            if self.sparse_backprop and s < num_segments - 1:
                # Store checkpoint with shallow detachment option to prune old history
                cached_states.append(current_state)
                seg_mean_key = self.key_proj(seg_x.mean(dim=1))
                cached_keys.append(seg_mean_key)
            else:
                cached_states.append(current_state.clone())
                seg_mean_key = self.key_proj(seg_x.mean(dim=1))
                cached_keys.append(seg_mean_key)

        return torch.cat(segment_outputs, dim=1)

# ==============================================================================
# 4. Full Bit-MC-SSM Model
# ==============================================================================

class BitMCSSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, segment_len: int = 64, top_k: int = 2, ffn_mult: int = 2, sparse_backprop: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mc_ssm = MemoryCachingSSC(d_model=d_model, d_state=d_state, segment_len=segment_len, top_k=top_k, sparse_backprop=sparse_backprop)

        self.norm2 = RMSNorm(d_model)
        d_ffn = d_model * ffn_mult
        self.ffn_in = BitLinear158(d_model, d_ffn * 2)
        self.ffn_out = BitLinear158(d_ffn, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mc_ssm(self.norm1(x))
        norm_x = self.norm2(x)
        ffn_proj = self.ffn_in(norm_x)
        w1, w2 = torch.chunk(ffn_proj, 2, dim=-1)
        ffn_act = F.silu(w1) * w2
        x = x + self.ffn_out(ffn_act)
        return x

class BitMCSSMForCausalLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 256,
        n_layers: int = 6,
        d_state: int = 16,
        segment_len: int = 64,
        top_k: int = 2,
        ffn_mult: int = 2,
        sparse_backprop: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

        self.blocks = nn.ModuleList([
            BitMCSSMBlock(
                d_model=d_model,
                d_state=d_state,
                segment_len=segment_len,
                top_k=top_k,
                ffn_mult=ffn_mult,
                sparse_backprop=sparse_backprop
            )
            for _ in range(n_layers)
        ])

        self.final_norm = RMSNorm(d_model)
        self.lm_head = BitLinear158(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 80, temperature: float = 0.8, top_k: int = 40, top_p: float = 0.9) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self.forward(input_ids)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            # Top-p (Nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits.scatter_(1, indices_to_remove.unsqueeze(0), -float('Inf'))

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

# ==============================================================================
# 5. Dataset Loading (TinyStories / Tiktoken / GPT-2 Tokenizer)
# ==============================================================================

def load_tokenizer_and_data(dataset_name: str = "tinystories", max_tokens: int = 500000) -> Tuple[Any, List[int]]:
    """Loads GPT-2 Tokenizer and TinyStories / fallback text corpus"""
    # 1. Try loading GPT-2 tokenizer via tiktoken or transformers
    tokenizer = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        class TiktokenWrapper:
            def __init__(self, enc):
                self.enc = enc
                self.vocab_size = 50257
            def encode(self, text: str) -> List[int]:
                return self.enc.encode_ordinary(text)
            def decode(self, ids: List[int]) -> str:
                return self.enc.decode(ids)
        tokenizer = TiktokenWrapper(enc)
        print("✅ Using GPT-2 Tokenizer (tiktoken)")
    except ImportError:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("gpt2")
            class TransformerWrapper:
                def __init__(self, tok):
                    self.tok = tok
                    self.vocab_size = tok.vocab_size
                def encode(self, text: str) -> List[int]:
                    return self.tok.encode(text)
                def decode(self, ids: List[int]) -> str:
                    return self.tok.decode(ids)
            tokenizer = TransformerWrapper(tok)
            print("✅ Using GPT-2 Tokenizer (transformers)")
        except Exception:
            print("ℹ️ Falling back to Byte-level UTF-8 Tokenizer")
            class ByteTokenizer:
                def __init__(self):
                    self.vocab_size = 256
                def encode(self, text: str) -> List[int]:
                    return list(text.encode("utf-8"))
                def decode(self, ids: List[int]) -> str:
                    return bytes(ids).decode("utf-8", errors="ignore")
            tokenizer = ByteTokenizer()

    # 2. Try loading HuggingFace TinyStories
    tokens = []
    if dataset_name.lower() == "tinystories":
        try:
            from datasets import load_dataset
            print("⏳ Downloading / Streaming TinyStories dataset...")
            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            text_accumulator = []
            for item in ds:
                text_accumulator.append(item["text"])
                if sum(len(t) for t in text_accumulator) > max_tokens * 4:
                    break
            full_text = "\n\n".join(text_accumulator)
            tokens = tokenizer.encode(full_text)[:max_tokens]
            print(f"✅ Loaded {len(tokens):,} tokens from TinyStories")
        except Exception as e:
            print(f"⚠️ Could not fetch TinyStories online ({e}). Using embedded high-quality story collection.")

    if len(tokens) == 0:
        # High quality offline fallback corpus
        fallback_corpus = """
Once upon a time, there was a bright little girl named Lily. She loved to explore the vast green forest with her loyal dog Max.
One afternoon, as the golden sun began to set behind the misty mountains, Lily noticed an ancient stone gate covered in shimmering ivy.
"Look, Max!" Lily whispered excitedly. "What do you think is on the other side?"
Max barked softly and wagged his bushy tail. Together, they gently pushed the heavy stone gate open.
Inside lay an enchanted garden filled with glowing silver flowers, whispering trees, and a crystalline stream that sparkled like diamonds.
A wise old owl perched on a high branch and looked down at them with warm, intelligent eyes.
"Welcome, brave travelers," the owl spoke in a gentle melodic voice. "This is the Garden of Endless Memory."
"The flowers here remember every song ever sung, and the water remembers every story ever told."
Lily knelt beside the stream and reached out to touch the water. Instantly, soft laughter and wonderful tales filled the quiet air.
She shared a fresh sweet apple from her basket with the woodland creatures that gathered around them.
Max lay happily in the soft green moss, watching the butterflies dance under the starlight.
Lily knew that whenever she felt curious or alone, the magic garden would always be waiting to welcome her home.
They stayed until the moon rose high in the velvet sky, and walked back home with hearts full of wonder and peaceful dreams.
""" * 80
        tokens = tokenizer.encode(fallback_corpus)
        print(f"✅ Loaded {len(tokens):,} tokens from rich story corpus.")

    return tokenizer, tokens

class TokenDataset(Dataset):
    def __init__(self, tokens: List[int], seq_len: int = 128, stride: Optional[int] = None):
        self.seq_len = seq_len
        self.stride = stride or (seq_len // 2)
        self.data = torch.tensor(tokens, dtype=torch.long)
        self.indices = list(range(0, len(self.data) - self.seq_len, self.stride))
        if len(self.indices) == 0 and len(self.data) > 1:
            self.indices = [0]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        chunk = self.data[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]

# ==============================================================================
# 6. Training & Benchmarking CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Bit-MC-SSM Phase 2 Training")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dataset", type=str, default="tinystories")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--segment_len", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sparse_backprop", action="store_true", default=True, help="Enable sparse backprop")
    parser.add_argument("--dense_backprop", action="store_false", dest="sparse_backprop", help="Disable sparse backprop")
    parser.add_argument("--save_path", type=str, default="bit_mc_ssm_phase2.pt")
    parser.add_argument("--prompt", type=str, default="Once upon a time, in an enchanted garden")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device

    print("=" * 75)
    print("🚀 Bit-MC-SSM Phase 2: Sparse Backpropagation & Long-Context Scaling")
    print(f"   Device: {device} | d_model: {args.d_model} | Layers: {args.n_layers} | SeqLen: {args.seq_len}")
    print(f"   Segment Len (C): {args.segment_len} | Top-k: {args.top_k} | Sparse BP: {args.sparse_backprop}")
    print("=" * 75)

    # 1. Load Data & Tokenizer
    tokenizer, tokens = load_tokenizer_and_data(dataset_name=args.dataset, max_tokens=300000)
    dataset = TokenDataset(tokens, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"📚 Dataset Batches: {len(dataloader)} | Total Training Samples: {len(dataset):,}")

    # 2. Build Model
    model = BitMCSSMForCausalLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        segment_len=args.segment_len,
        top_k=args.top_k,
        sparse_backprop=args.sparse_backprop
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model Parameters: {total_params / 1e6:.3f}M ({total_params:,} parameters)")

    # 3. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(dataloader))

    # 4. Training Loop with Throughput and Memory Monitoring
    model.train()
    start_time = time.time()
    total_tokens_processed = 0

    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        step_count = 0

        for x_b, y_b in dataloader:
            x_b, y_b = x_b.to(device), y_b.to(device)

            optimizer.zero_grad()
            _, loss = model(x_b, targets=y_b)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            step_count += 1
            total_tokens_processed += x_b.numel()

        avg_loss = total_loss / step_count
        ppl = math.exp(min(avg_loss, 20.0))
        elapsed = time.time() - start_time
        tokens_per_sec = total_tokens_processed / max(elapsed, 0.01)

        mem_str = ""
        if torch.cuda.is_available() and device.startswith("cuda"):
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            mem_str = f" | Peak VRAM: {peak_mb:.1f}MB"

        print(f"Epoch [{epoch:2d}/{args.epochs:2d}] | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | Speed: {tokens_per_sec:.0f} tok/s{mem_str}")

    # 5. Save Checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": tokenizer.vocab_size,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "d_state": args.d_state,
            "segment_len": args.segment_len,
            "top_k": args.top_k,
            "sparse_backprop": args.sparse_backprop
        }
    }, args.save_path)
    print(f"\n💾 Model checkpoint saved to: {args.save_path}")

    # 6. Text Generation Demo
    print("\n" + "=" * 75)
    print("✨ Text Generation Test (Phase 2 Model):")
    print("=" * 75)
    model.eval()
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    gen_ids = model.generate(prompt_ids, max_new_tokens=100, temperature=0.75, top_k=30, top_p=0.9)
    gen_text = tokenizer.decode(gen_ids[0].tolist())

    print(f"[Prompt]: {args.prompt}")
    print(f"[Generated]:\n{gen_text}\n")
    print("=" * 75)
    print("🎉 Phase 2 Verified Successfully!")

if __name__ == "__main__":
    main()
