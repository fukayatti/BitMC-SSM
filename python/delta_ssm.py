"""
Delta-SSM / Dual-State (Fast + Slow) Transition Engine
Inspired by RWKV-6, DeltaNet, and Eagle architectures.

Features:
1. Delta-Gated State Transitions: Updates state only when delta |Delta h_t| exceeds threshold epsilon.
   Skips redundant memory store operations on CPU.
2. Dual-State Architecture:
   - Fast State (h_fast): Short-term syntactic details & immediate dependencies.
   - Slow State (h_slow): Long-term semantic topic & document persistence.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeltaSSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, delta_thresh: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.delta_thresh = delta_thresh
        
        # Projections
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.b_fast = nn.Linear(d_model, d_state, bias=False)
        self.b_slow = nn.Linear(d_model, d_state, bias=False)
        
        self.c_fast = nn.Linear(d_state, d_model, bias=False)
        self.c_slow = nn.Linear(d_state, d_model, bias=False)
        
        # Learnable Fast & Slow Decays
        # Fast decay: ~ 0.5 (fast forgetting)
        # Slow decay: ~ 0.98 (long-term preservation)
        self.decay_fast = nn.Parameter(torch.sigmoid(torch.linspace(-0.5, 0.5, d_state)))
        self.decay_slow = nn.Parameter(torch.sigmoid(torch.linspace(2.5, 4.5, d_state)))
        
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward_step(self, x_t: torch.Tensor, h_fast_prev: torch.Tensor, h_slow_prev: torch.Tensor):
        """
        Step-wise forward inference with delta gating (O(1) memory & store skip).
        x_t: (B, d_model)
        h_fast_prev, h_slow_prev: (B, d_state)
        """
        # Input gating
        proj = self.in_proj(x_t)
        u, gate = proj.chunk(2, dim=-1)
        u_act = u * F.silu(gate)
        
        # 1. Fast State Transition
        b_f = self.b_fast(u_act)
        delta_f = (self.decay_fast - 1.0) * h_fast_prev + b_f
        # Apply deadband delta threshold during inference
        mask_f = (torch.abs(delta_f) > self.delta_thresh).float()
        h_fast = h_fast_prev + mask_f * delta_f
        
        # 2. Slow State Transition
        b_s = self.b_slow(u_act)
        delta_s = (self.decay_slow - 1.0) * h_slow_prev + b_s
        mask_s = (torch.abs(delta_s) > self.delta_thresh).float()
        h_slow = h_slow_prev + mask_s * delta_s
        
        # Output aggregation
        y_fast = self.c_fast(h_fast)
        y_slow = self.c_slow(h_slow)
        y = self.out_proj(y_fast + y_slow + u_act)
        
        # Return updated states and skipped update ratio
        skip_ratio = 1.0 - (mask_f.mean().item() + mask_s.mean().item()) / 2.0
        return y, h_fast, h_slow, skip_ratio

    def forward_sequence(self, x_seq: torch.Tensor):
        """
        Full sequence forward pass.
        x_seq: (B, L, d_model)
        """
        B, L, D = x_seq.shape
        h_fast = torch.zeros(B, self.d_state, device=x_seq.device)
        h_slow = torch.zeros(B, self.d_state, device=x_seq.device)
        
        outputs = []
        total_skips = 0.0
        for t in range(L):
            y_t, h_fast, h_slow, skip_r = self.forward_step(x_seq[:, t], h_fast, h_slow)
            outputs.append(y_t)
            total_skips += skip_r
            
        avg_skip_ratio = total_skips / L
        return torch.stack(outputs, dim=1), avg_skip_ratio
