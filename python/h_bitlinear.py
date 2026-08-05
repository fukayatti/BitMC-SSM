"""
BitNet v2: Native 4-bit Activations with Hadamard Transformation (H-BitLinear)
Reference: "BitNet v2: Native 4-bit Activations with Hadamard Transformation for 1-bit LLMs"
           Microsoft Research (arXiv:2504.18415v2, April 2025)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def fast_hadamard_transform(x: torch.Tensor, scale: float = None) -> torch.Tensor:
    """
    Computes Fast Walsh-Hadamard Transform (FWHT) along the last dimension.
    Complexity: O(n log n) operations using ONLY additions and subtractions.
    """
    d = x.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)

    # Next power of 2
    next_pow2 = 1 << (d - 1).bit_length()
    if next_pow2 != d:
        x_pad = F.pad(x, (0, next_pow2 - d))
    else:
        x_pad = x

    shape = x_pad.shape
    curr = x_pad.reshape(-1, next_pow2)
    h = 1
    while h < next_pow2:
        curr = curr.view(-1, next_pow2 // (2 * h), 2, h)
        a = curr[:, :, 0, :]
        b = curr[:, :, 1, :]
        curr = torch.stack([a + b, a - b], dim=2)
        h *= 2

    curr = curr.reshape(shape)
    if next_pow2 != d:
        curr = curr[..., :d]

    return curr * scale


class QuantizeAct4Bit(torch.autograd.Function):
    """
    4-bit (INT4: -8 to +7) Activation Quantization with Straight-Through Estimator (STE)
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float = 1e-5):
        # Per-token absmean scaling as recommended in BitNet v2
        gamma = x.abs().mean(dim=-1, keepdim=True).clamp(min=eps)
        scale = 7.0 / (gamma * 1.5)  # Scale such that 1.5 * absmean maps to 7
        x_scaled = x * scale
        x_q = torch.clamp(torch.round(x_scaled), -8.0, 7.0) / scale
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class QuantizeWeightTernary(torch.autograd.Function):
    """
    1.58-bit Ternary Weight Quantization {-1, 0, +1} with optional Deadband Sparsity
    """
    @staticmethod
    def forward(ctx, weight: torch.Tensor, tau: float = 0.85, eps: float = 1e-5):
        gamma = weight.abs().mean().clamp(min=eps)
        w_scaled = weight / gamma
        w_q = torch.zeros_like(w_scaled)
        w_q = torch.where(w_scaled > tau, 1.0, w_q)
        w_q = torch.where(w_scaled < -tau, -1.0, w_q)
        return w_q * gamma

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


# Import Triton fused kernel suite
try:
    from .triton_kernels import fused_hadamard_act4, fused_ternary_quant
except ImportError:
    try:
        from triton_kernels import fused_hadamard_act4, fused_ternary_quant
    except ImportError:
        fused_hadamard_act4 = None
        fused_ternary_quant = None


class HBitLinear(nn.Linear):
    """
    H-BitLinear Layer (BitNet v2)
    Applies Fast Walsh-Hadamard Transform (FWHT) prior to 4-bit activation quantization,
    effectively suppressing outlier channels into a Gaussian distribution.
    Accelerated with Triton GPU kernels where available.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, tau: float = 0.85, use_hadamard: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        self.tau = tau
        self.use_hadamard = use_hadamard
        self.hadamard_scale = 1.0 / math.sqrt(in_features)

        # Kaiming-style initialization
        nn.init.normal_(self.weight, std=math.sqrt(2.0 / (in_features + out_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1 & 2. Fused Hadamard Transform + 4-bit Activation Quantization
        if fused_hadamard_act4 is not None:
            x_q = fused_hadamard_act4(x, use_hadamard=self.use_hadamard)
        else:
            if self.use_hadamard:
                x_h = fast_hadamard_transform(x, scale=self.hadamard_scale)
            else:
                x_h = x
            x_q = QuantizeAct4Bit.apply(x_h)

        # 3. 1.58-bit Ternary Weight Quantization
        if fused_ternary_quant is not None:
            w_q = fused_ternary_quant(self.weight, self.tau)
        else:
            w_q = QuantizeWeightTernary.apply(self.weight, self.tau)

        # 4. Multiply (Zero-GEMM in C++ engine)
        return F.linear(x_q, w_q, self.bias)

    def get_sparsity_stats(self) -> dict:
        with torch.no_grad():
            gamma = self.weight.abs().mean().clamp(min=1e-5)
            w_scaled = self.weight / gamma
            zero_mask = (w_scaled.abs() <= self.tau)
            return {
                "zero_weight_ratio": zero_mask.float().mean().item(),
                "in_features": self.in_features,
                "out_features": self.out_features,
                "tau": self.tau
            }
