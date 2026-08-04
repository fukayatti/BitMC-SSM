"""
Bit-MC-SSM (1.58-bit Memory-Cached State Space Model)
Phase 1: PyTorch Reference Implementation & Training Script
"""

import math
import time
import argparse
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. Quantization & 1.58-bit BitLinear (with STE: Straight-Through Estimator)
# ==============================================================================

def weight_quant(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Ternary weight quantization to {-1, 0, +1} using BitNet b1.58 formula.
    W_quant = clip(round(W / gamma), -1, 1) * gamma
    """
    gamma = torch.mean(torch.abs(w)) + eps
    w_scaled = w / gamma
    w_quant = torch.clamp(torch.round(w_scaled), -1.0, 1.0) * gamma
    # STE: forward uses w_quant, backward passes gradients to w
    return w + (w_quant - w).detach()

def activation_quant(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    8-bit activation quantization to [-128, 127].
    X_quant = clip(round(X * 127 / Q_b), -128, 127) * (Q_b / 127)
    """
    scale = 127.0 / (torch.max(torch.abs(x), dim=-1, keepdim=True)[0] + eps)
    x_scaled = x * scale
    x_quant = torch.clamp(torch.round(x_scaled), -128.0, 127.0) / scale
    # STE
    return x + (x_quant - x).detach()

class BitLinear158(nn.Linear):
    """
    BitNet 1.58-bit Linear Layer.
    Weights in {-1, 0, +1}, Activations in INT8.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        # Latent continuous weights (initialized with scaled normal)
        nn.init.normal_(self.weight, std=math.sqrt(2.0 / (in_features + out_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Quantize activation
        x_q = activation_quant(x)
        # 2. Quantize weight
        w_q = weight_quant(self.weight)
        # 3. Linear projection (During Phase 3 C++, this becomes integer addition/subtraction)
        return F.linear(x_q, w_q, self.bias)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (with INT8 compatibility)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        return self.weight * x_norm

# ==============================================================================
# 2. Shift-SSM: Bit-Shift State Space Recurrent Layer
# ==============================================================================

class ShiftSSM(nn.Module):
    """
    Shift-SSM: Discrete-friendly State Space Model.
    - Intra-segment token mixing with 1D Depthwise Conv
    - Recurrent state decay approximated via bit-shift friendly dynamics (2^-s)
    - 1.58-bit projections for input/output/gates
    """
    def __init__(self, d_model: int, d_state: int = 16, conv_kernel: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.conv_kernel = conv_kernel

        # Projections using 1.58-bit BitLinear
        self.in_proj = BitLinear158(d_model, d_model * 2) # [u, gate]
        self.b_proj = BitLinear158(d_model, d_state)     # Input state projection B
        self.c_proj = BitLinear158(d_model, d_state)     # Output state projection C
        self.out_proj = BitLinear158(d_model, d_model)

        # 1D Depthwise Conv for short-term local context
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=d_model
        )

        # Logarithmic decay parameter (decay = sigmoid(dt_log) ~ 2^-s)
        self.decay_param = nn.Parameter(torch.randn(d_state) * 0.1 - 1.0)

    def forward(self, x: torch.Tensor, initial_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]
            initial_state: [batch, d_state]
        Returns:
            y: [batch, seq_len, d_model]
            final_state: [batch, d_state]
        """
        b, l, d = x.shape

        # 1. In projection & gating split
        projected = self.in_proj(x) # [b, l, 2*d]
        u, gate = torch.chunk(projected, 2, dim=-1)

        # 2. Local 1D Conv (causal)
        u_conv = self.conv1d(u.transpose(1, 2))[:, :, :l].transpose(1, 2)
        u_conv = F.silu(u_conv)

        # 3. B, C projections
        B = self.b_proj(u_conv) # [b, l, d_state]
        C = self.c_proj(u_conv) # [b, l, d_state]

        # 4. Shift Decay (differentiable decay factor between 0 and 1)
        decay = torch.sigmoid(self.decay_param).unsqueeze(0).unsqueeze(0) # [1, 1, d_state]

        # 5. Recurrent State Update
        h_t = initial_state if initial_state is not None else torch.zeros(b, self.d_state, device=x.device, dtype=x.dtype)
        state_outputs = []

        for t in range(l):
            # h_t = decay * h_{t-1} + B_t * u_conv_t_mean
            u_t = u_conv[:, t, :].mean(dim=-1, keepdim=True) # scalar per token for state update
            b_t = B[:, t, :]
            h_t = decay.squeeze(1) * h_t + b_t * u_t
            
            # y_t = (C_t * h_t).sum() projected back to d_model
            c_t = C[:, t, :]
            state_val = (c_t * h_t).sum(dim=-1, keepdim=True) # [b, 1]
            state_outputs.append(state_val)

        state_seq = torch.cat(state_outputs, dim=-1).unsqueeze(-1) # [b, l, 1]
        
        # Merge local representation with state representation
        mixed = u_conv + state_seq.expand(-1, -1, d)
        gated = mixed * F.silu(gate)
        out = self.out_proj(gated)

        return out, h_t

# ==============================================================================
# 3. Sparse Selective Memory Caching (MC-SSC, arXiv:2602.24281)
# ==============================================================================

class MemoryCachingSSC(nn.Module):
    """
    Memory Caching with Sparse Selective Caching (SSC).
    - Splits sequence into segments of length C (e.g., 64, 128)
    - Caches segment checkpoints M^(s) and pooled segment keys
    - Query selectively routes to Top-k relevant past checkpoints
    - Gated residual aggregation
    """
    def __init__(self, d_model: int, d_state: int = 16, segment_len: int = 64, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.segment_len = segment_len
        self.top_k = top_k

        self.ssm = ShiftSSM(d_model=d_model, d_state=d_state)
        self.query_proj = BitLinear158(d_model, d_model)
        self.key_proj = BitLinear158(d_model, d_model)
        self.gate_proj = BitLinear158(d_model, 2) # [gamma_online, gamma_cached]
        self.cache_out_proj = BitLinear158(d_state, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            out: [batch, seq_len, d_model]
        """
        b, l, d = x.shape
        num_segments = math.ceil(l / self.segment_len)

        segment_outputs = []
        cached_states: List[torch.Tensor] = []     # List of [b, d_state]
        cached_keys: List[torch.Tensor] = []       # List of [b, d_model]

        current_state = None

        for s in range(num_segments):
            start_idx = s * self.segment_len
            end_idx = min((s + 1) * self.segment_len, l)
            seg_x = x[:, start_idx:end_idx, :]
            seg_len = end_idx - start_idx

            # 1. Process Segment with Local SSM
            ssm_out, current_state = self.ssm(seg_x, initial_state=current_state)

            # 2. If we have past cached checkpoints, apply Sparse Selective Retrieval
            if len(cached_states) > 0 and self.top_k > 0:
                # Queries for this segment
                queries = self.query_proj(seg_x) # [b, seg_len, d]

                # Stack cached keys: [b, num_cached, d]
                k_stack = torch.stack(cached_keys, dim=1)
                # Stack cached states: [b, num_cached, d_state]
                s_stack = torch.stack(cached_states, dim=1)

                # Compute similarity scores: [b, seg_len, num_cached]
                # Cosine-like dot product
                scores = torch.einsum("btd, bcd -> btc", F.normalize(queries, dim=-1), F.normalize(k_stack, dim=-1))

                # Top-k selection
                k_val = min(self.top_k, len(cached_states))
                topk_scores, topk_indices = torch.topk(scores, k=k_val, dim=-1) # [b, seg_len, k]
                topk_weights = F.softmax(topk_scores, dim=-1)                   # [b, seg_len, k]

                # Gather top-k states and aggregate
                # Expand s_stack for batch gathering: [b, seg_len, k, d_state]
                topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_state)
                s_stack_expanded = s_stack.unsqueeze(1).expand(-1, seg_len, -1, -1)
                gathered_states = torch.gather(s_stack_expanded, 2, topk_indices_expanded)

                # Weighted sum of retrieved states: [b, seg_len, d_state]
                retrieved_state = (gathered_states * topk_weights.unsqueeze(-1)).sum(dim=2)
                retrieved_info = self.cache_out_proj(retrieved_state) # [b, seg_len, d]

                # Gated Aggregation
                gates = F.softmax(self.gate_proj(seg_x), dim=-1) # [b, seg_len, 2]
                g_online, g_cached = gates[..., 0:1], gates[..., 1:2]

                seg_out = g_online * ssm_out + g_cached * retrieved_info
            else:
                seg_out = ssm_out

            segment_outputs.append(seg_out)

            # 3. Checkpoint the segment state & mean key
            cached_states.append(current_state.clone())
            seg_mean_key = self.key_proj(seg_x.mean(dim=1)) # [b, d]
            cached_keys.append(seg_mean_key)

        return torch.cat(segment_outputs, dim=1)

# ==============================================================================
# 4. Full Transformer-Free Bit-MC-SSM Model
# ==============================================================================

class BitMCSSMBlock(nn.Module):
    """Single Bit-MC-SSM Transformer-free Block"""
    def __init__(self, d_model: int, d_state: int = 16, segment_len: int = 64, top_k: int = 2, ffn_mult: int = 2):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mc_ssm = MemoryCachingSSC(d_model=d_model, d_state=d_state, segment_len=segment_len, top_k=top_k)

        self.norm2 = RMSNorm(d_model)
        # SwiGLU / BitLinear MLP
        d_ffn = d_model * ffn_mult
        self.ffn_in = BitLinear158(d_model, d_ffn * 2)
        self.ffn_out = BitLinear158(d_ffn, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. SSM + MC Residual
        x = x + self.mc_ssm(self.norm1(x))
        # 2. BitLinear SwiGLU FFN Residual
        norm_x = self.norm2(x)
        ffn_proj = self.ffn_in(norm_x)
        w1, w2 = torch.chunk(ffn_proj, 2, dim=-1)
        ffn_act = F.silu(w1) * w2
        x = x + self.ffn_out(ffn_act)
        return x

class BitMCSSMForCausalLM(nn.Module):
    """
    Bit-MC-SSM Causal Language Model
    Complete GPU-less & Multiplication-free inference ready architecture.
    """
    def __init__(
        self,
        vocab_size: int = 4096,
        d_model: int = 256,
        n_layers: int = 6,
        d_state: int = 16,
        segment_len: int = 64,
        top_k: int = 2,
        ffn_mult: int = 2
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token Embedding (Lookup Table)
        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

        # Blocks
        self.blocks = nn.ModuleList([
            BitMCSSMBlock(
                d_model=d_model,
                d_state=d_state,
                segment_len=segment_len,
                top_k=top_k,
                ffn_mult=ffn_mult
            )
            for _ in range(n_layers)
        ])

        self.final_norm = RMSNorm(d_model)
        # Unembedding Head (BitLinear 1.58b)
        self.lm_head = BitLinear158(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.embedding(input_ids) # [b, l, d]

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x) # [b, l, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50, temperature: float = 0.8, top_k: int = 40) -> torch.Tensor:
        """Autoregressive text generation"""
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self.forward(input_ids)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

# ==============================================================================
# 5. Dataset & Simple Tokenizer (Self-Contained Fallback)
# ==============================================================================

class SimpleCharTokenizer:
    """Lightweight character/byte-level tokenizer for standalone execution"""
    def __init__(self, text_corpus: str):
        chars = sorted(list(set(text_corpus)))
        self.vocab_size = max(len(chars) + 2, 256)
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.itos = {i + 1: ch for i, ch in enumerate(chars)}
        self.stoi['<pad>'] = 0
        self.itos[0] = '<pad>'
        self.stoi['<unk>'] = len(self.stoi)
        self.itos[self.stoi['<unk>']] = '<unk>'

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(c, self.stoi['<unk>']) for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join([self.itos.get(i, "") for i in ids if i != 0])

class TextDataset(Dataset):
    def __init__(self, token_ids: List[int], seq_len: int = 128, stride: Optional[int] = None):
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.indices = list(range(0, len(self.data) - self.seq_len, self.stride))
        if len(self.indices) == 0 and len(self.data) > 1:
            self.indices = [0]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.indices[idx]
        chunk = self.data[start : start + self.seq_len + 1]
        if len(chunk) < self.seq_len + 1:
            # Pad if needed
            pad_len = self.seq_len + 1 - len(chunk)
            chunk = torch.cat([chunk, torch.zeros(pad_len, dtype=torch.long)])
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def get_demo_corpus() -> str:
    """Sample story corpus for immediate validation without external network dependency"""
    return """
Once upon a time, there was a little girl named Lily. She loved to explore the sunny garden behind her house.
One bright morning, Lily saw a tiny green frog sitting near a big round pond.
"Hello, little frog!" Lily said with a warm smile.
The frog hopped happily onto a smooth stone and made a soft sound.
Lily found a shiny red apple under a tree. She wanted to share something sweet with her new friend.
She placed a small piece of fruit near the stone.
The frog looked at Lily, blinked its big black eyes, and jumped gently into the cool blue water.
Lily laughed and watched the ripples spread across the pond.
Every afternoon, Lily came back to the garden to tell stories to the frog.
They became the best of friends, and the garden was always filled with joy and sunshine.
Once upon a time, a small brown puppy named Max found a lost kitten in the quiet forest.
Max barked gently and guided the kitten safely back home.
The little girl gave Max a tasty biscuit and hugged the kitten tightly.
Everyone was very happy and they slept peacefully all night long.
""" * 50 # Repeat to provide sufficient text

# ==============================================================================
# 6. Training & Validation Execution
# ==============================================================================

def train():
    parser = argparse.ArgumentParser(description="Bit-MC-SSM Phase 1 Training")
    parser.add_argument("--device", type=str, default="auto", help="cuda, cpu, mps or auto")
    parser.add_argument("--d_model", type=int, default=128, help="Model hidden dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--d_state", type=int, default=16, help="SSM State dimension")
    parser.add_argument("--segment_len", type=int, default=32, help="Memory caching segment length C")
    parser.add_argument("--top_k", type=int, default=2, help="Sparse routing Top-k cached states")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--prompt", type=str, default="Once upon a time, Lily saw", help="Generation test prompt")
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device

    print("=" * 70)
    print("🚀 Bit-MC-SSM Phase 1: Prototype Training & Simulation")
    print(f"   Device: {device} | d_model: {args.d_model} | Layers: {args.n_layers}")
    print(f"   Segment Len (C): {args.segment_len} | Top-k: {args.top_k} | State Dim: {args.d_state}")
    print("=" * 70)

    # 1. Prepare Data & Tokenizer
    corpus = get_demo_corpus()
    tokenizer = SimpleCharTokenizer(corpus)
    token_ids = tokenizer.encode(corpus)
    dataset = TextDataset(token_ids, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"📚 Dataset: {len(token_ids)} tokens | Vocab Size: {tokenizer.vocab_size} | Batches: {len(dataloader)}")

    # 2. Instantiate Model
    model = BitMCSSMForCausalLM(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        segment_len=args.segment_len,
        top_k=args.top_k
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Total Parameters: {total_params / 1e6:.3f}M ({total_params:,} params)")

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(dataloader))

    # 4. Training Loop
    model.train()
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        step_count = 0

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            _, loss = model(x_batch, targets=y_batch)
            loss.backward()

            # Gradient clipping for stable STE training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            step_count += 1

        avg_loss = total_loss / step_count
        ppl = math.exp(min(avg_loss, 20.0))
        elapsed = time.time() - start_time
        print(f"Epoch [{epoch:2d}/{args.epochs:2d}] | Loss: {avg_loss:.4f} | Perplexity: {ppl:.2f} | Time: {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("✨ Training Completed! Running Text Generation Test...")
    print("=" * 70)

    # 5. Generation Demo
    model.eval()
    prompt_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    generated_ids = model.generate(prompt_ids, max_new_tokens=150, temperature=0.7, top_k=20)
    generated_text = tokenizer.decode(generated_ids[0].tolist())

    print(f"\n[Prompt]: {args.prompt}")
    print(f"[Generated]:\n{generated_text}\n")
    print("=" * 70)
    print("🎉 Phase 1 Prototype Verified Successfully!")

if __name__ == "__main__":
    train()
