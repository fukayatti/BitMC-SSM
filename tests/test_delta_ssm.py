"""
Test Suite for Delta-SSM / Dual-State Transition Engine
Validates:
1. Memory update skip ratio (skips redundant memory stores).
2. Sequence preservation & gradient flow.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from python.delta_ssm import DeltaSSMBlock


def test_delta_ssm_skip_and_learning():
    torch.manual_seed(42)
    B, L, D = 4, 64, 64
    d_state = 32
    
    layer = DeltaSSMBlock(d_model=D, d_state=d_state, delta_thresh=0.02)
    
    # Input with some redundant consecutive tokens (e.g. repeated patterns / whitespace)
    x = torch.randn(B, L, D)
    x[:, 10:20] = x[:, 9:10] # repeated tokens
    
    out, skip_ratio = layer.forward_sequence(x)
    
    print(f"   [Delta-SSM] Output shape: {out.shape}")
    print(f"   🚀 State Update Memory Write Skip Ratio: {skip_ratio * 100:.2f}% (Memory bus bandwidth saved!)")
    assert out.shape == (B, L, D)
    assert skip_ratio > 0.15, "Expected significant update skipping on repeated tokens"
    
    # Verify backward pass / gradient flow
    loss = out.sum()
    loss.backward()
    assert layer.decay_fast.grad is not None
    assert layer.decay_slow.grad is not None
    print("   ✅ Delta-SSM gradient flow & update skipping PASSED!")


if __name__ == "__main__":
    print("=" * 70)
    print("⚡ Running Delta-SSM / Dual-State Test Suite")
    print("=" * 70)
    test_delta_ssm_skip_and_learning()
    print("=" * 70)
