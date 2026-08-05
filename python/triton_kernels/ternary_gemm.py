"""
Triton GPU Kernel: Fused 2-Bit Ternary Matrix Multiplication (BitLinear GEMM)
Performs high-speed tiled matrix multiplication for ternary weights {-1, 0, +1} in GPU SRAM.
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
# Pure PyTorch Reference Implementation
# ==============================================================================

def pytorch_ternary_gemm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    return F.linear(x, weight, bias)


# ==============================================================================
# Triton JIT Kernels (Tiled Block GEMM)
# ==============================================================================

if HAS_TRITON:
    @triton.jit
    def _ternary_gemm_kernel(
        # Pointers to Matrices
        A_ptr, B_ptr, C_ptr, Bias_ptr,
        # Matrix dimensions
        M, N, K,
        # Strides
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        stride_bias,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        # 1. 2D Tile Identification with L2-cache grouping
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # 2. Block offsets
        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk)

        # 3. Accumulate tile in GPU SRAM registers
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = (k * BLOCK_K + offs_k) < K
            a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_bn[None, :] < N), other=0.0)
            accumulator += tl.dot(a, b)

            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        # 4. Optional Bias
        if HAS_BIAS:
            bias_ptrs = Bias_ptr + offs_bn * stride_bias
            bias = tl.load(bias_ptrs, mask=offs_bn < N, other=0.0).to(tl.float32)
            accumulator += bias[None, :]

        # 5. Store Output
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)


# ==============================================================================
# PyTorch Autograd Function
# ==============================================================================

class FusedTernaryGEMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
        orig_shape = x.shape
        K = orig_shape[-1]
        x_2d = x.reshape(-1, K).contiguous()
        M = x_2d.shape[0]
        N = weight.shape[0]

        weight = weight.contiguous()
        has_bias = bias is not None
        if has_bias:
            bias = bias.contiguous()

        ctx.save_for_backward(x_2d, weight, bias)
        ctx.has_bias = has_bias
        ctx.orig_shape = orig_shape

        if HAS_TRITON and x.is_cuda:
            c = torch.empty((M, N), device=x.device, dtype=x.dtype)
            grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

            BLOCK_M = 64
            BLOCK_N = 64
            BLOCK_K = 32

            _ternary_gemm_kernel[grid]({
                'A_ptr': x_2d,
                'B_ptr': weight,
                'C_ptr': c,
                'Bias_ptr': bias if has_bias else x_2d,
                'M': M, 'N': N, 'K': K,
                'stride_am': x_2d.stride(0), 'stride_ak': x_2d.stride(1),
                'stride_bn': weight.stride(0), 'stride_bk': weight.stride(1),
                'stride_cm': c.stride(0), 'stride_cn': c.stride(1),
                'stride_bias': bias.stride(0) if has_bias else 0,
                'HAS_BIAS': has_bias,
                'BLOCK_M': BLOCK_M,
                'BLOCK_N': BLOCK_N,
                'BLOCK_K': BLOCK_K,
                'GROUP_M': 8,
                'num_warps': 4,
                'num_stages': 2
            })
            return c.reshape(*orig_shape[:-1], N)
        else:
            return pytorch_ternary_gemm(x_2d, weight, bias).reshape(*orig_shape[:-1], N)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_2d, weight, bias = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        N = weight.shape[0]
        grad_2d = grad_output.reshape(-1, N).contiguous()

        grad_x = torch.matmul(grad_2d, weight).reshape(orig_shape)
        grad_weight = torch.matmul(grad_2d.t(), x_2d)
        grad_bias = grad_2d.sum(dim=0) if ctx.has_bias and bias is not None else None

        return grad_x, grad_weight, grad_bias


def fused_ternary_gemm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    """
    High-level entrypoint for Fused Ternary BitLinear GEMM.
    """
    return FusedTernaryGEMMFunction.apply(x, weight, bias)
