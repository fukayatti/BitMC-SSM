"""
Unit tests and parity verification for all Triton GPU kernels and fallback implementations:
- Fused FWHT + INT4 Quantization
- Fused Delta-SSM Recurrent State Scan
- Fused 1.58-bit Ternary Weight Deadband Quantization
- Fused Flash Cross-Entropy Loss
- Fused RMSNorm & Fused SiLU Gating
- Fused 2-Bit Ternary GEMM
"""

import math
import pytest
import torch
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from triton_kernels import (
    HAS_TRITON,
    fused_hadamard_act4,
    fused_delta_ssm_scan,
    fused_ternary_quant,
    fused_cross_entropy,
    fused_silu_gating,
    fused_ternary_gemm,
    FusedRMSNorm,
    pytorch_fast_hadamard_transform,
    pytorch_quantize_act_4bit,
    pytorch_delta_ssm_scan,
    pytorch_quantize_weight_ternary,
    pytorch_cross_entropy,
    pytorch_rmsnorm,
    pytorch_silu_gating,
    pytorch_ternary_gemm,
)


def test_hadamard_act4_forward_parity():
    torch.manual_seed(42)
    B, L, D = 2, 16, 64
    x = torch.randn(B, L, D, requires_grad=True)

    # 1. PyTorch Reference
    x_h = pytorch_fast_hadamard_transform(x, scale=1.0 / math.sqrt(D))
    y_ref = pytorch_quantize_act_4bit(x_h)

    # 2. Fused Implementation
    y_fused = fused_hadamard_act4(x, use_hadamard=True)

    assert y_fused.shape == (B, L, D)
    assert torch.allclose(y_ref, y_fused, atol=1e-5), "Hadamard + Act4 forward outputs must match reference"


def test_hadamard_act4_backward_ste():
    torch.manual_seed(42)
    B, L, D = 2, 8, 32
    x = torch.randn(B, L, D, requires_grad=True)

    y = fused_hadamard_act4(x, use_hadamard=True)
    loss = (y * 2.0).sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (B, L, D)
    assert not torch.isnan(x.grad).any()


def test_delta_ssm_scan_forward_backward_parity():
    torch.manual_seed(42)
    B, L, S = 2, 16, 32
    x_in = torch.randn(B, L, S, requires_grad=True)
    decay = torch.sigmoid(torch.tensor([-2.0] * S)).requires_grad_(True)
    C_t = torch.randn(B, L, S, requires_grad=True)

    # Reference PyTorch Scan
    y_ref, next_state_ref = pytorch_delta_ssm_scan(x_in, decay, C_t)
    loss_ref = y_ref.sum()
    loss_ref.backward()
    grad_x_ref = x_in.grad.clone()
    grad_d_ref = decay.grad.clone()
    grad_c_ref = C_t.grad.clone()

    # Reset grads
    x_in.grad.zero_()
    decay.grad.zero_()
    C_t.grad.zero_()

    # Fused Scan
    y_fused, next_state_fused = fused_delta_ssm_scan(x_in, decay, C_t)
    loss_fused = y_fused.sum()
    loss_fused.backward()

    assert torch.allclose(y_ref, y_fused, atol=1e-5), "Delta-SSM scan y_t must match reference"
    assert torch.allclose(next_state_ref, next_state_fused, atol=1e-5), "Delta-SSM scan next_state must match reference"
    assert torch.allclose(grad_x_ref, x_in.grad, atol=1e-4), "Delta-SSM grad_x must match reference"
    assert torch.allclose(grad_c_ref, C_t.grad, atol=1e-4), "Delta-SSM grad_C must match reference"


def test_ternary_weight_quant_parity():
    torch.manual_seed(42)
    W = torch.randn(128, 64, requires_grad=True)
    tau = 0.85

    w_ref = pytorch_quantize_weight_ternary(W, tau=tau)
    w_fused = fused_ternary_quant(W, tau=tau)

    assert torch.allclose(w_ref, w_fused, atol=1e-5), "Ternary weight quantization must match reference"

    # STE gradient
    loss = (w_fused * 1.5).sum()
    loss.backward()
    assert W.grad is not None
    assert torch.allclose(W.grad, torch.ones_like(W) * 1.5, atol=1e-5)


def test_cross_entropy_forward_backward_parity():
    torch.manual_seed(42)
    N, V = 16, 128
    logits = torch.randn(N, V, requires_grad=True)
    targets = torch.randint(0, V, (N,))

    # Reference
    loss_ref = pytorch_cross_entropy(logits, targets)
    loss_ref.backward()
    grad_ref = logits.grad.clone()

    logits.grad.zero_()

    # Fused
    loss_fused = fused_cross_entropy(logits, targets)
    loss_fused.backward()
    grad_fused = logits.grad.clone()

    assert torch.allclose(loss_ref, loss_fused, atol=1e-5), "Cross-entropy loss must match reference"
    assert torch.allclose(grad_ref, grad_fused, atol=1e-4), "Cross-entropy gradients must match reference"


def test_rmsnorm_forward_backward_parity():
    torch.manual_seed(42)
    B, L, D = 2, 8, 64
    x = torch.randn(B, L, D, requires_grad=True)
    weight = torch.ones(D, requires_grad=True)

    # Reference
    y_ref = pytorch_rmsnorm(x, weight)
    loss_ref = y_ref.sum()
    loss_ref.backward()
    grad_x_ref = x.grad.clone()
    grad_w_ref = weight.grad.clone()

    x.grad.zero_()
    weight.grad.zero_()

    # Fused Module
    norm = FusedRMSNorm(D)
    norm.weight.data.copy_(weight.data)
    y_fused = norm(x)
    loss_fused = y_fused.sum()
    loss_fused.backward()

    assert torch.allclose(y_ref, y_fused, atol=1e-5), "RMSNorm forward output must match reference"
    assert torch.allclose(grad_x_ref, x.grad, atol=1e-4), "RMSNorm grad_x must match reference"
    assert torch.allclose(grad_w_ref, norm.weight.grad, atol=1e-4), "RMSNorm grad_weight must match reference"


def test_silu_gating_parity():
    torch.manual_seed(42)
    B, L, D = 2, 8, 64
    f1 = torch.randn(B, L, D, requires_grad=True)
    f2 = torch.randn(B, L, D, requires_grad=True)

    # Reference
    y_ref = pytorch_silu_gating(f1, f2)
    loss_ref = y_ref.sum()
    loss_ref.backward()
    grad_f1_ref = f1.grad.clone()
    grad_f2_ref = f2.grad.clone()

    f1.grad.zero_()
    f2.grad.zero_()

    # Fused
    y_fused = fused_silu_gating(f1, f2)
    loss_fused = y_fused.sum()
    loss_fused.backward()

    assert torch.allclose(y_ref, y_fused, atol=1e-5), "SiLU Gating output must match reference"
    assert torch.allclose(grad_f1_ref, f1.grad, atol=1e-4), "SiLU Gating grad_f1 must match reference"
    assert torch.allclose(grad_f2_ref, f2.grad, atol=1e-4), "SiLU Gating grad_f2 must match reference"


def test_ternary_gemm_forward_backward_parity():
    torch.manual_seed(42)
    M, K, N = 16, 64, 32
    x = torch.randn(M, K, requires_grad=True)
    weight = torch.randn(N, K, requires_grad=True)
    bias = torch.randn(N, requires_grad=True)

    # Reference
    y_ref = pytorch_ternary_gemm(x, weight, bias)
    loss_ref = y_ref.sum()
    loss_ref.backward()
    grad_x_ref = x.grad.clone()
    grad_w_ref = weight.grad.clone()
    grad_b_ref = bias.grad.clone()

    x.grad.zero_()
    weight.grad.zero_()
    bias.grad.zero_()

    # Fused GEMM
    y_fused = fused_ternary_gemm(x, weight, bias)
    loss_fused = y_fused.sum()
    loss_fused.backward()

    assert torch.allclose(y_ref, y_fused, atol=1e-5), "Ternary GEMM forward output must match reference"
    assert torch.allclose(grad_x_ref, x.grad, atol=1e-4), "Ternary GEMM grad_x must match reference"
    assert torch.allclose(grad_w_ref, weight.grad, atol=1e-4), "Ternary GEMM grad_w must match reference"
    assert torch.allclose(grad_b_ref, bias.grad, atol=1e-4), "Ternary GEMM grad_bias must match reference"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
