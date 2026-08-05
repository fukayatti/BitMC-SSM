"""
Triton GPU Kernel: Fused Flash Cross-Entropy Loss
Computes Cross-Entropy loss directly in GPU SRAM with online LogSumExp tracking,
avoiding huge DRAM allocations and accelerating vocabulary projection losses.
"""

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

def pytorch_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)


# ==============================================================================
# Triton JIT Kernels (Online LogSumExp Forward & Fused Backward)
# ==============================================================================

if HAS_TRITON:
    @triton.jit
    def _cross_entropy_fwd_kernel(
        Logits_ptr, Targets_ptr, Losses_ptr, LSE_ptr,
        stride_ln, stride_lv,
        stride_t, stride_loss, stride_lse,
        N, V,
        ignore_index,
        BLOCK_V: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        if row_idx >= N:
            return

        target = tl.load(Targets_ptr + row_idx * stride_t)
        if target == ignore_index:
            tl.store(Losses_ptr + row_idx * stride_loss, 0.0)
            tl.store(LSE_ptr + row_idx * stride_lse, 0.0)
            return

        # Phase 1: Online Max & Sum of Exponentials
        m_curr = -float('inf')
        s_curr = 0.0

        for off_v in range(0, V, BLOCK_V):
            offs = off_v + tl.arange(0, BLOCK_V)
            mask = offs < V
            logits = tl.load(Logits_ptr + row_idx * stride_ln + offs * stride_lv, mask=mask, other=-float('inf')).to(tl.float32)

            m_chunk = tl.max(logits, axis=0)
            m_new = tl.maximum(m_curr, m_chunk)
            # Re-scale running sum with difference in max
            s_curr = s_curr * tl.exp(m_curr - m_new) + tl.sum(tl.exp(logits - m_new), axis=0)
            m_curr = m_new

        lse = m_curr + tl.log(s_curr)
        tl.store(LSE_ptr + row_idx * stride_lse, lse)

        # Phase 2: Target Logit & Loss
        target_logit = tl.load(Logits_ptr + row_idx * stride_ln + target * stride_lv).to(tl.float32)
        loss = lse - target_logit
        tl.store(Losses_ptr + row_idx * stride_loss, loss)


    @triton.jit
    def _cross_entropy_bwd_kernel(
        Grad_Logits_ptr, Logits_ptr, Targets_ptr, LSE_ptr,
        stride_gln, stride_glv,
        stride_ln, stride_lv,
        stride_t, stride_lse,
        N, V,
        grad_scale,
        ignore_index,
        BLOCK_V: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        if row_idx >= N:
            return

        target = tl.load(Targets_ptr + row_idx * stride_t)
        if target == ignore_index:
            for off_v in range(0, V, BLOCK_V):
                offs = off_v + tl.arange(0, BLOCK_V)
                mask = offs < V
                tl.store(Grad_Logits_ptr + row_idx * stride_gln + offs * stride_glv, 0.0, mask=mask)
            return

        lse = tl.load(LSE_ptr + row_idx * stride_lse)

        for off_v in range(0, V, BLOCK_V):
            offs = off_v + tl.arange(0, BLOCK_V)
            mask = offs < V
            logits = tl.load(Logits_ptr + row_idx * stride_ln + offs * stride_lv, mask=mask, other=-float('inf')).to(tl.float32)

            # Softmax probability: exp(logit - lse)
            probs = tl.exp(logits - lse)
            # Subtract 1.0 for the target index
            is_target = (offs == target)
            grad_val = tl.where(is_target, probs - 1.0, probs) * grad_scale

            tl.store(Grad_Logits_ptr + row_idx * stride_gln + offs * stride_glv, grad_val, mask=mask)


# ==============================================================================
# PyTorch Autograd Function
# ==============================================================================

class FusedCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100):
        N, V = logits.shape
        logits = logits.contiguous()
        targets = targets.contiguous()

        if HAS_TRITON and logits.is_cuda:
            losses = torch.empty((N,), device=logits.device, dtype=torch.float32)
            lse = torch.empty((N,), device=logits.device, dtype=torch.float32)

            BLOCK_V = 1024
            grid = (N,)
            _cross_entropy_fwd_kernel[grid](
                logits, targets, losses, lse,
                logits.stride(0), logits.stride(1),
                targets.stride(0), losses.stride(0), lse.stride(0),
                N, V,
                ignore_index,
                BLOCK_V=BLOCK_V,
                num_warps=4
            )
            ctx.save_for_backward(logits, targets, lse)
            ctx.ignore_index = ignore_index
            ctx.N = N
            ctx.V = V
            ctx.is_triton = True
            return losses.mean()
        else:
            loss = pytorch_cross_entropy(logits, targets, ignore_index=ignore_index)
            ctx.save_for_backward(logits, targets)
            ctx.ignore_index = ignore_index
            ctx.is_triton = False
            return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if not getattr(ctx, "is_triton", False):
            logits, targets = ctx.saved_tensors
            with torch.enable_grad():
                l_req = logits.detach().requires_grad_(True)
                loss = pytorch_cross_entropy(l_req, targets, ignore_index=ctx.ignore_index)
                loss.backward(grad_output)
            return l_req.grad, None, None

        logits, targets, lse = ctx.saved_tensors
        N, V = ctx.N, ctx.V
        grad_scale = (grad_output / N).item()

        grad_logits = torch.empty_like(logits)
        BLOCK_V = 1024
        grid = (N,)

        _cross_entropy_bwd_kernel[grid](
            grad_logits, logits, targets, lse,
            grad_logits.stride(0), grad_logits.stride(1),
            logits.stride(0), logits.stride(1),
            targets.stride(0), lse.stride(0),
            N, V,
            grad_scale,
            ctx.ignore_index,
            BLOCK_V=BLOCK_V,
            num_warps=4
        )

        return grad_logits, None, None


def fused_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """
    High-level entrypoint for Fused Flash Cross-Entropy Loss.
    """
    return FusedCrossEntropyFunction.apply(logits, targets, ignore_index)
