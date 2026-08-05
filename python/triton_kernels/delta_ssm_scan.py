"""
Triton GPU Kernel: Fused Delta-SSM Recurrent State Scan
Fuses the recurrent state update (h_t = decay * h_{t-1} + x_t) and output projection (y_t = C_t * h_t)
into GPU SRAM registers, avoiding O(L^2) memory footprint and DRAM latency.
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
# Pure PyTorch Reference Implementations (Vectorized Einsum Scan & Fallback)
# ==============================================================================

def pytorch_delta_ssm_scan(x_in: torch.Tensor, decay: torch.Tensor, C_t: torch.Tensor):
    """
    x_in:  [B, L, S]
    decay: [S]
    C_t:   [B, L, S]
    Returns:
      y_t:        [B, L, 1]
      next_state: [B, S]
    """
    B, L, S = x_in.shape
    t_idx = torch.arange(L, device=x_in.device)
    diff = (t_idx[:, None] - t_idx[None, :]).clamp(min=0)
    mask = (t_idx[:, None] >= t_idx[None, :]).float()
    decay_mat = (decay.view(1, 1, -1) ** diff.unsqueeze(-1)) * mask.unsqueeze(-1)
    h_seq = torch.einsum('l k s, b k s -> b l s', decay_mat, x_in)
    y_t = (C_t * h_seq).sum(dim=-1, keepdim=True)
    next_state = h_seq[:, -1, :]
    return y_t, next_state


# ==============================================================================
# Triton JIT Kernels (Forward & Backward)
# ==============================================================================

if HAS_TRITON:
    @triton.jit
    def _delta_ssm_scan_fwd_kernel(
        X_in_ptr, Decay_ptr, C_t_ptr,
        Y_ptr, Next_State_ptr, H_Seq_ptr,
        stride_xb, stride_xl, stride_xs,
        stride_cb, stride_cl, stride_cs,
        stride_yb, stride_yl,
        stride_hb, stride_hl, stride_hs,
        stride_nb, stride_ns,
        B, L, S,
        SAVE_H_SEQ: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        b_idx = tl.program_id(0)
        if b_idx >= B:
            return

        offs_s = tl.arange(0, BLOCK_S)
        mask_s = offs_s < S

        # Load decay [S]
        decay_vals = tl.load(Decay_ptr + offs_s, mask=mask_s, other=0.0)

        # Initialize state h in GPU thread registers
        h = tl.zeros((BLOCK_S,), dtype=tl.float32)

        for t in range(L):
            # 1. Load x_in[b, t, :]
            x_ptr = X_in_ptr + b_idx * stride_xb + t * stride_xl + offs_s * stride_xs
            x_val = tl.load(x_ptr, mask=mask_s, other=0.0).to(tl.float32)

            # 2. State update: h = decay * h + x
            h = decay_vals * h + x_val

            # Optional save of h_seq for backward pass
            if SAVE_H_SEQ:
                h_ptr = H_Seq_ptr + b_idx * stride_hb + t * stride_hl + offs_s * stride_hs
                tl.store(h_ptr, h, mask=mask_s)

            # 3. Load C_t[b, t, :]
            c_ptr = C_t_ptr + b_idx * stride_cb + t * stride_cl + offs_s * stride_cs
            c_val = tl.load(c_ptr, mask=mask_s, other=0.0).to(tl.float32)

            # 4. y_t = sum(C_t * h)
            y_val = tl.sum(tl.where(mask_s, c_val * h, 0.0), axis=0)

            # 5. Store y_t[b, t, 0]
            y_ptr = Y_ptr + b_idx * stride_yb + t * stride_yl
            tl.store(y_ptr, y_val)

        # Store next_state[b, :]
        next_ptr = Next_State_ptr + b_idx * stride_nb + offs_s * stride_ns
        tl.store(next_ptr, h, mask=mask_s)


    @triton.jit
    def _delta_ssm_scan_bwd_kernel(
        Grad_Y_ptr, Decay_ptr, C_t_ptr, H_Seq_ptr,
        Grad_X_ptr, Grad_C_ptr, Grad_Decay_Accum_ptr,
        stride_gyb, stride_gyl,
        stride_cb, stride_cl, stride_cs,
        stride_hb, stride_hl, stride_hs,
        stride_gxb, stride_gxl, stride_gxs,
        stride_gcb, stride_gcl, stride_gcs,
        stride_gdb, stride_gds,
        B, L, S,
        BLOCK_S: tl.constexpr,
    ):
        b_idx = tl.program_id(0)
        if b_idx >= B:
            return

        offs_s = tl.arange(0, BLOCK_S)
        mask_s = offs_s < S

        decay_vals = tl.load(Decay_ptr + offs_s, mask=mask_s, other=0.0)

        # Running gradient w.r.t state: grad_h
        grad_h = tl.zeros((BLOCK_S,), dtype=tl.float32)
        grad_decay_accum = tl.zeros((BLOCK_S,), dtype=tl.float32)

        # Traverse time in reverse: L-1 down to 0
        for step in range(L):
            t = L - 1 - step

            # Load grad_y[b, t]
            gy_ptr = Grad_Y_ptr + b_idx * stride_gyb + t * stride_gyl
            gy = tl.load(gy_ptr).to(tl.float32)

            # Load C_t[b, t, :]
            c_ptr = C_t_ptr + b_idx * stride_cb + t * stride_cl + offs_s * stride_cs
            c_val = tl.load(c_ptr, mask=mask_s, other=0.0).to(tl.float32)

            # Load h_seq[b, t, :]
            h_ptr = H_Seq_ptr + b_idx * stride_hb + t * stride_hl + offs_s * stride_hs
            h_t = tl.load(h_ptr, mask=mask_s, other=0.0).to(tl.float32)

            # 1. grad_C_t = grad_y * h_t
            grad_c = gy * h_t
            gc_ptr = Grad_C_ptr + b_idx * stride_gcb + t * stride_gcl + offs_s * stride_gcs
            tl.store(gc_ptr, grad_c, mask=mask_s)

            # 2. Update grad_h: grad_h = gy * C_t + grad_h (from next timestep)
            grad_h = grad_h + gy * c_val

            # 3. grad_x[b, t] = grad_h
            gx_ptr = Grad_X_ptr + b_idx * stride_gxb + t * stride_gxl + offs_s * stride_gxs
            tl.store(gx_ptr, grad_h, mask=mask_s)

            # 4. Accumulate grad_decay: grad_decay += grad_h * h_{t-1}
            if t > 0:
                h_prev_ptr = H_Seq_ptr + b_idx * stride_hb + (t - 1) * stride_hl + offs_s * stride_hs
                h_prev = tl.load(h_prev_ptr, mask=mask_s, other=0.0).to(tl.float32)
                grad_decay_accum += grad_h * h_prev

            # 5. Propagate grad_h to previous timestep: grad_h = grad_h * decay
            grad_h = grad_h * decay_vals

        # Store per-batch grad_decay contribution
        gd_ptr = Grad_Decay_Accum_ptr + b_idx * stride_gdb + offs_s * stride_gds
        tl.store(gd_ptr, grad_decay_accum, mask=mask_s)


# ==============================================================================
# PyTorch Autograd Function
# ==============================================================================

class FusedDeltaSSMScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in: torch.Tensor, decay: torch.Tensor, C_t: torch.Tensor):
        B, L, S = x_in.shape
        x_in = x_in.contiguous()
        decay = decay.contiguous()
        C_t = C_t.contiguous()

        if HAS_TRITON and x_in.is_cuda:
            BLOCK_S = triton.next_power_of_2(S)
            y = torch.empty((B, L, 1), device=x_in.device, dtype=x_in.dtype)
            next_state = torch.empty((B, S), device=x_in.device, dtype=x_in.dtype)
            h_seq = torch.empty((B, L, S), device=x_in.device, dtype=torch.float32)

            grid = (B,)
            _delta_ssm_scan_fwd_kernel[grid](
                x_in, decay, C_t,
                y, next_state, h_seq,
                x_in.stride(0), x_in.stride(1), x_in.stride(2),
                C_t.stride(0), C_t.stride(1), C_t.stride(2),
                y.stride(0), y.stride(1),
                h_seq.stride(0), h_seq.stride(1), h_seq.stride(2),
                next_state.stride(0), next_state.stride(1),
                B, L, S,
                SAVE_H_SEQ=True,
                BLOCK_S=BLOCK_S,
                num_warps=2
            )
            ctx.save_for_backward(decay, C_t, h_seq)
            ctx.B, ctx.L, ctx.S = B, L, S
            ctx.BLOCK_S = BLOCK_S
            return y, next_state
        else:
            y, next_state = pytorch_delta_ssm_scan(x_in, decay, C_t)
            ctx.save_for_backward(x_in, decay, C_t)
            ctx.is_pytorch_fallback = True
            return y, next_state

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor, grad_next_state: torch.Tensor = None):
        if getattr(ctx, "is_pytorch_fallback", False):
            # Autograd fallback using computational graph
            x_in, decay, C_t = ctx.saved_tensors
            with torch.enable_grad():
                x_req = x_in.detach().requires_grad_(True)
                d_req = decay.detach().requires_grad_(True)
                c_req = C_t.detach().requires_grad_(True)
                y, _ = pytorch_delta_ssm_scan(x_req, d_req, c_req)
                y.backward(grad_y)
            return x_req.grad, d_req.grad, c_req.grad

        decay, C_t, h_seq = ctx.saved_tensors
        B, L, S = ctx.B, ctx.L, ctx.S
        BLOCK_S = ctx.BLOCK_S
        grad_y = grad_y.contiguous()

        grad_x = torch.empty((B, L, S), device=decay.device, dtype=decay.dtype)
        grad_c = torch.empty((B, L, S), device=decay.device, dtype=decay.dtype)
        grad_decay_batch = torch.empty((B, S), device=decay.device, dtype=torch.float32)

        grid = (B,)
        _delta_ssm_scan_bwd_kernel[grid](
            grad_y, decay, C_t, h_seq,
            grad_x, grad_c, grad_decay_batch,
            grad_y.stride(0), grad_y.stride(1),
            C_t.stride(0), C_t.stride(1), C_t.stride(2),
            h_seq.stride(0), h_seq.stride(1), h_seq.stride(2),
            grad_x.stride(0), grad_x.stride(1), grad_x.stride(2),
            grad_c.stride(0), grad_c.stride(1), grad_c.stride(2),
            grad_decay_batch.stride(0), grad_decay_batch.stride(1),
            B, L, S,
            BLOCK_S=BLOCK_S,
            num_warps=2
        )

        grad_decay = grad_decay_batch.sum(dim=0).to(decay.dtype)
        return grad_x, grad_decay, grad_c


def fused_delta_ssm_scan(x_in: torch.Tensor, decay: torch.Tensor, C_t: torch.Tensor):
    """
    High-level entrypoint for Fused Delta-SSM Recurrent State Scan.
    """
    return FusedDeltaSSMScanFunction.apply(x_in, decay, C_t)
