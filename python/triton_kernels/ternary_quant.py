"""
Triton GPU Kernel: Fused 1.58-bit Ternary Weight Deadband Quantization
Implements high-speed {-1, 0, +1} weight quantization with Straight-Through Estimator (STE).
"""

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ==============================================================================
# Pure PyTorch Reference Implementation
# ==============================================================================

def pytorch_quantize_weight_ternary(weight: torch.Tensor, tau: float = 0.85, eps: float = 1e-5) -> torch.Tensor:
    gamma = weight.abs().mean().clamp(min=eps)
    w_scaled = weight / gamma
    w_q = torch.zeros_like(w_scaled)
    w_q = torch.where(w_scaled > tau, 1.0, w_q)
    w_q = torch.where(w_scaled < -tau, -1.0, w_q)
    return w_q * gamma


# ==============================================================================
# Triton JIT Kernels
# ==============================================================================

if HAS_TRITON:
    @triton.jit
    def _ternary_quant_kernel(
        W_ptr, W_Q_ptr,
        num_elements,
        gamma,
        tau,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < num_elements

        w = tl.load(W_ptr + offs, mask=mask, other=0.0)
        w_scaled = w / gamma

        # Ternary {-1, 0, +1} thresholding
        pos_mask = w_scaled > tau
        neg_mask = w_scaled < -tau

        q_val = tl.where(pos_mask, 1.0, tl.where(neg_mask, -1.0, 0.0))
        w_q = q_val * gamma

        tl.store(W_Q_ptr + offs, w_q, mask=mask)


# ==============================================================================
# PyTorch Autograd Function
# ==============================================================================

class FusedTernaryQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, tau: float = 0.85, eps: float = 1e-5):
        if HAS_TRITON and weight.is_cuda:
            gamma = weight.abs().mean().clamp(min=eps).item()
            num_elements = weight.numel()
            w_q = torch.empty_like(weight)

            BLOCK_SIZE = 1024
            grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
            _ternary_quant_kernel[grid](
                weight, w_q,
                num_elements,
                gamma,
                tau,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4
            )
            return w_q
        else:
            return pytorch_quantize_weight_ternary(weight, tau=tau, eps=eps)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE: Gradients propagate directly through ternary quantization
        return grad_output, None, None


def fused_ternary_quant(weight: torch.Tensor, tau: float = 0.85, eps: float = 1e-5) -> torch.Tensor:
    """
    High-level entrypoint for Fused Ternary Weight Deadband Quantization with STE.
    """
    return FusedTernaryQuantFunction.apply(weight, tau, eps)
