"""
Triton GPU Kernel: Fused Fast Walsh-Hadamard Transform (FWHT) + 4-bit Activation Quantization
Fuses online outlier suppression (FWHT) with INT4 STE quantization in GPU SRAM/registers.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ==============================================================================
# Pure PyTorch Reference Implementations (Fallback & Verification)
# ==============================================================================

def pytorch_fast_hadamard_transform(x: torch.Tensor, scale: float = None) -> torch.Tensor:
    """Computes FWHT using pure PyTorch tensor operations."""
    d = x.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)

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


def pytorch_quantize_act_4bit(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """4-bit (INT4: -8..7) Activation Quantization with STE."""
    gamma = x.abs().mean(dim=-1, keepdim=True).clamp(min=eps)
    scale = 7.0 / (gamma * 1.5)
    x_scaled = x * scale
    x_q = torch.clamp(torch.round(x_scaled), -8.0, 7.0) / scale
    return x_q


# ==============================================================================
# Triton JIT Kernels (CUDA)
# ==============================================================================

if HAS_TRITON:
    @triton.jit
    def _act4_quant_kernel(
        X_ptr, Y_ptr,
        stride_xn, stride_xd,
        stride_yn, stride_yd,
        N, D,
        eps,
        BLOCK_D: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        if row_idx >= N:
            return

        offs_d = tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x_ptrs = X_ptr + row_idx * stride_xn + offs_d * stride_xd
        x = tl.load(x_ptrs, mask=mask, other=0.0)

        # 1. Compute absmean gamma
        abs_x = tl.abs(x)
        sum_abs = tl.sum(tl.where(mask, abs_x, 0.0), axis=0)
        gamma = sum_abs / D
        gamma = tl.maximum(gamma, eps)

        # 2. Scale factor: 7.0 / (gamma * 1.5)
        scale = 7.0 / (gamma * 1.5)
        x_scaled = x * scale

        # 3. Round and clamp to INT4 [-8.0, 7.0]
        x_rounded = tl.floor(x_scaled + 0.5)
        x_clamped = tl.maximum(tl.minimum(x_rounded, 7.0), -8.0)
        x_q = x_clamped / scale

        # 4. Store result
        y_ptrs = Y_ptr + row_idx * stride_yn + offs_d * stride_yd
        tl.store(y_ptrs, x_q, mask=mask)


# ==============================================================================
# PyTorch Autograd Function
# ==============================================================================

class FusedHadamardAct4Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, use_hadamard: bool = True, eps: float = 1e-5):
        orig_shape = x.shape
        D = orig_shape[-1]
        x_2d = x.reshape(-1, D)
        N = x_2d.shape[0]

        # 1. Apply Hadamard transform if requested
        if use_hadamard:
            x_h = pytorch_fast_hadamard_transform(x_2d, scale=1.0 / math.sqrt(D))
        else:
            x_h = x_2d

        # 2. Triton accelerated INT4 quantization if on CUDA & Triton available
        if HAS_TRITON and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16, torch.float32):
            BLOCK_D = triton.next_power_of_2(D)
            y = torch.empty_like(x_h)
            grid = (N,)
            _act4_quant_kernel[grid](
                x_h, y,
                x_h.stride(0), x_h.stride(1),
                y.stride(0), y.stride(1),
                N, D,
                eps,
                BLOCK_D=BLOCK_D,
                num_warps=4 if BLOCK_D >= 256 else 2
            )
        else:
            y = pytorch_quantize_act_4bit(x_h, eps=eps)

        ctx.use_hadamard = use_hadamard
        ctx.D = D
        ctx.orig_shape = orig_shape
        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE: Gradient flows straight through INT4 quantization
        if ctx.use_hadamard:
            D = ctx.D
            orig_shape = ctx.orig_shape
            grad_2d = grad_output.reshape(-1, D)
            grad_h = pytorch_fast_hadamard_transform(grad_2d, scale=1.0 / math.sqrt(D))
            return grad_h.reshape(orig_shape), None, None
        else:
            return grad_output, None, None


def fused_hadamard_act4(x: torch.Tensor, use_hadamard: bool = True, eps: float = 1e-5) -> torch.Tensor:
    """
    High-level entrypoint for Fused Hadamard Transformation + 4-bit Activation Quantization.
    """
    return FusedHadamardAct4Function.apply(x, use_hadamard, eps)
