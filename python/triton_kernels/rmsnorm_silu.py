"""
Triton GPU Kernel: Fused RMSNorm & Fused SiLU Gating
Fuses root-mean-square normalization and SiLU gating (SwiGLU-style) into single SRAM GPU operations
with full forward and backward autograd support.
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
# Pure PyTorch Reference Implementations
# ==============================================================================

def pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def pytorch_silu_gating(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
    return F.silu(f1) * f2


# ==============================================================================
# Triton JIT Kernels (RMSNorm & SiLU Gating Forward & Backward)
# ==============================================================================

if HAS_TRITON:
    @triton.jit
    def _rmsnorm_fwd_kernel(
        X_ptr, Weight_ptr, Y_ptr, Rsqrt_ptr,
        stride_xn, stride_xd,
        stride_w,
        stride_yn, stride_yd,
        stride_rsqrt,
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
        w_ptrs = Weight_ptr + offs_d * stride_w

        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptrs, mask=mask, other=1.0).to(tl.float32)

        # 1. Variance = mean(x^2)
        var = tl.sum(x * x, axis=0) / D
        rsqrt = 1.0 / tl.sqrt(var + eps)
        tl.store(Rsqrt_ptr + row_idx * stride_rsqrt, rsqrt)

        # 2. Output = x * rsqrt * w
        y = x * rsqrt * w
        y_ptrs = Y_ptr + row_idx * stride_yn + offs_d * stride_yd
        tl.store(y_ptrs, y, mask=mask)


    @triton.jit
    def _rmsnorm_bwd_kernel(
        Grad_Y_ptr, X_ptr, Weight_ptr, Rsqrt_ptr,
        Grad_X_ptr, Grad_Weight_Accum_ptr,
        stride_gyn, stride_gyd,
        stride_xn, stride_xd,
        stride_w, stride_rsqrt,
        stride_gxn, stride_gxd,
        stride_gwn, stride_gwd,
        N, D,
        BLOCK_D: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        if row_idx >= N:
            return

        offs_d = tl.arange(0, BLOCK_D)
        mask = offs_d < D

        gy_ptrs = Grad_Y_ptr + row_idx * stride_gyn + offs_d * stride_gyd
        x_ptrs = X_ptr + row_idx * stride_xn + offs_d * stride_xd
        w_ptrs = Weight_ptr + offs_d * stride_w

        gy = tl.load(gy_ptrs, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptrs, mask=mask, other=1.0).to(tl.float32)
        rsqrt = tl.load(Rsqrt_ptr + row_idx * stride_rsqrt).to(tl.float32)

        # 1. grad_weight row contribution
        gw_row = gy * (x * rsqrt)
        gw_ptrs = Grad_Weight_Accum_ptr + row_idx * stride_gwn + offs_d * stride_gwd
        tl.store(gw_ptrs, gw_row, mask=mask)

        # 2. grad_x
        gy_w = gy * w
        dot = tl.sum(gy_w * x, axis=0)
        gx = rsqrt * (gy_w - (x * (rsqrt * rsqrt) / D) * dot)

        gx_ptrs = Grad_X_ptr + row_idx * stride_gxn + offs_d * stride_gxd
        tl.store(gx_ptrs, gx, mask=mask)


    @triton.jit
    def _silu_gating_fwd_kernel(
        F1_ptr, F2_ptr, Y_ptr,
        stride_f1n, stride_f1d,
        stride_f2n, stride_f2d,
        stride_yn, stride_yd,
        N, D,
        BLOCK_D: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        if row_idx >= N:
            return

        offs_d = tl.arange(0, BLOCK_D)
        mask = offs_d < D

        f1 = tl.load(F1_ptr + row_idx * stride_f1n + offs_d * stride_f1d, mask=mask, other=0.0).to(tl.float32)
        f2 = tl.load(F2_ptr + row_idx * stride_f2n + offs_d * stride_f2d, mask=mask, other=0.0).to(tl.float32)

        # silu(f1) = f1 * sigmoid(f1)
        sig = tl.sigmoid(f1)
        silu_f1 = f1 * sig
        y = silu_f1 * f2

        y_ptrs = Y_ptr + row_idx * stride_yn + offs_d * stride_yd
        tl.store(y_ptrs, y, mask=mask)


    @triton.jit
    def _silu_gating_bwd_kernel(
        Grad_Y_ptr, F1_ptr, F2_ptr,
        Grad_F1_ptr, Grad_F2_ptr,
        stride_gyn, stride_gyd,
        stride_f1n, stride_f1d,
        stride_f2n, stride_f2d,
        stride_gf1n, stride_gf1d,
        stride_gf2n, stride_gf2d,
        N, D,
        BLOCK_D: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        if row_idx >= N:
            return

        offs_d = tl.arange(0, BLOCK_D)
        mask = offs_d < D

        gy = tl.load(Grad_Y_ptr + row_idx * stride_gyn + offs_d * stride_gyd, mask=mask, other=0.0).to(tl.float32)
        f1 = tl.load(F1_ptr + row_idx * stride_f1n + offs_d * stride_f1d, mask=mask, other=0.0).to(tl.float32)
        f2 = tl.load(F2_ptr + row_idx * stride_f2n + offs_d * stride_f2d, mask=mask, other=0.0).to(tl.float32)

        sig1 = tl.sigmoid(f1)
        silu1 = f1 * sig1
        dsilu1 = sig1 + f1 * sig1 * (1.0 - sig1)

        gf1 = gy * f2 * dsilu1
        gf2 = gy * silu1

        tl.store(Grad_F1_ptr + row_idx * stride_gf1n + offs_d * stride_gf1d, gf1, mask=mask)
        tl.store(Grad_F2_ptr + row_idx * stride_gf2n + offs_d * stride_gf2d, gf2, mask=mask)


# ==============================================================================
# PyTorch Autograd Functions
# ==============================================================================

class FusedRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
        orig_shape = x.shape
        D = orig_shape[-1]
        x_2d = x.reshape(-1, D).contiguous()
        weight = weight.contiguous()
        N = x_2d.shape[0]

        if HAS_TRITON and x.is_cuda:
            BLOCK_D = triton.next_power_of_2(D)
            y = torch.empty_like(x_2d)
            rsqrt = torch.empty((N,), device=x.device, dtype=torch.float32)

            grid = (N,)
            _rmsnorm_fwd_kernel[grid](
                x_2d, weight, y, rsqrt,
                x_2d.stride(0), x_2d.stride(1),
                weight.stride(0),
                y.stride(0), y.stride(1),
                rsqrt.stride(0),
                N, D,
                eps,
                BLOCK_D=BLOCK_D,
                num_warps=4 if BLOCK_D >= 256 else 2
            )
            ctx.save_for_backward(x_2d, weight, rsqrt)
            ctx.orig_shape = orig_shape
            ctx.D = D
            ctx.N = N
            ctx.BLOCK_D = BLOCK_D
            ctx.is_triton = True
            return y.reshape(orig_shape)
        else:
            y = pytorch_rmsnorm(x_2d, weight, eps=eps)
            ctx.save_for_backward(x_2d, weight)
            ctx.eps = eps
            ctx.orig_shape = orig_shape
            ctx.is_triton = False
            return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not getattr(ctx, "is_triton", False):
            x_2d, weight = ctx.saved_tensors
            with torch.enable_grad():
                x_req = x_2d.detach().requires_grad_(True)
                w_req = weight.detach().requires_grad_(True)
                y = pytorch_rmsnorm(x_req, w_req, eps=ctx.eps)
                y.backward(grad_output.reshape_as(y))
            return x_req.grad.reshape(ctx.orig_shape), w_req.grad, None

        x_2d, weight, rsqrt = ctx.saved_tensors
        D = ctx.D
        N = ctx.N
        BLOCK_D = ctx.BLOCK_D

        grad_2d = grad_output.reshape(-1, D).contiguous()
        grad_x = torch.empty_like(x_2d)
        grad_weight_accum = torch.empty((N, D), device=x_2d.device, dtype=torch.float32)

        grid = (N,)
        _rmsnorm_bwd_kernel[grid](
            grad_2d, x_2d, weight, rsqrt,
            grad_x, grad_weight_accum,
            grad_2d.stride(0), grad_2d.stride(1),
            x_2d.stride(0), x_2d.stride(1),
            weight.stride(0), rsqrt.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            grad_weight_accum.stride(0), grad_weight_accum.stride(1),
            N, D,
            BLOCK_D=BLOCK_D,
            num_warps=4 if BLOCK_D >= 256 else 2
        )

        grad_weight = grad_weight_accum.sum(dim=0).to(weight.dtype)
        return grad_x.reshape(ctx.orig_shape), grad_weight, None


class FusedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return FusedRMSNormFunction.apply(x, self.weight, self.eps)


class FusedSiLUGatingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f1: torch.Tensor, f2: torch.Tensor):
        orig_shape = f1.shape
        D = orig_shape[-1]
        f1_2d = f1.reshape(-1, D).contiguous()
        f2_2d = f2.reshape(-1, D).contiguous()
        N = f1_2d.shape[0]

        ctx.save_for_backward(f1_2d, f2_2d)
        ctx.orig_shape = orig_shape
        ctx.D = D
        ctx.N = N

        if HAS_TRITON and f1.is_cuda:
            BLOCK_D = triton.next_power_of_2(D)
            y = torch.empty_like(f1_2d)
            grid = (N,)
            _silu_gating_fwd_kernel[grid](
                f1_2d, f2_2d, y,
                f1_2d.stride(0), f1_2d.stride(1),
                f2_2d.stride(0), f2_2d.stride(1),
                y.stride(0), y.stride(1),
                N, D,
                BLOCK_D=BLOCK_D,
                num_warps=4 if BLOCK_D >= 256 else 2
            )
            ctx.BLOCK_D = BLOCK_D
            ctx.is_triton = True
            return y.reshape(orig_shape)
        else:
            ctx.is_triton = False
            return (F.silu(f1_2d) * f2_2d).reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        f1_2d, f2_2d = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        D = ctx.D
        N = ctx.N

        if not getattr(ctx, "is_triton", False):
            with torch.enable_grad():
                f1_req = f1_2d.detach().requires_grad_(True)
                f2_req = f2_2d.detach().requires_grad_(True)
                y = F.silu(f1_req) * f2_req
                y.backward(grad_output.reshape_as(y))
            return f1_req.grad.reshape(orig_shape), f2_req.grad.reshape(orig_shape)

        grad_2d = grad_output.reshape(-1, D).contiguous()
        grad_f1 = torch.empty_like(f1_2d)
        grad_f2 = torch.empty_like(f2_2d)
        BLOCK_D = ctx.BLOCK_D

        grid = (N,)
        _silu_gating_bwd_kernel[grid](
            grad_2d, f1_2d, f2_2d,
            grad_f1, grad_f2,
            grad_2d.stride(0), grad_2d.stride(1),
            f1_2d.stride(0), f1_2d.stride(1),
            f2_2d.stride(0), f2_2d.stride(1),
            grad_f1.stride(0), grad_f1.stride(1),
            grad_f2.stride(0), grad_f2.stride(1),
            N, D,
            BLOCK_D=BLOCK_D,
            num_warps=4 if BLOCK_D >= 256 else 2
        )

        return grad_f1.reshape(orig_shape), grad_f2.reshape(orig_shape)


def fused_silu_gating(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
    """
    High-level entrypoint for Fused SiLU Gating with autograd support.
    """
    return FusedSiLUGatingFunction.apply(f1, f2)
