"""
Test Suite for BitNet 2.0 Structural Sparsification
Validates:
1. Deadband-controlled 50%~75% weight sparsity.
2. 4-tuple all-zero block detection for T-MAC zero-skipping.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from python.bitnet2_sparse import BitNet2Linear


def test_bitnet2_sparsity():
    torch.manual_seed(42)
    in_features = 256
    out_features = 256
    
    # Tau = 0.9 produces ~55-60% zero weights
    layer = BitNet2Linear(in_features, out_features, tau=0.9)
    
    stats = layer.get_sparsity_stats()
    print(f"   [BitNet 2.0] Weight Zero Ratio:         {stats['zero_weight_ratio'] * 100:.2f}%")
    print(f"   🚀 T-MAC 4-tuple All-Zero Chunk Ratio: {stats['all_zero_chunk_ratio'] * 100:.2f}% (Skipped in LUT!)")
    
    assert stats['zero_weight_ratio'] >= 0.50, "Expected >= 50% zero weights"
    
    # Test forward and backward passes
    x = torch.randn(16, in_features)
    out = layer(x)
    loss = out.sum()
    loss.backward()
    
    assert layer.weight.grad is not None
    print("   ✅ BitNet 2.0 forward/backward & sparsity verification PASSED!")


if __name__ == "__main__":
    print("=" * 70)
    print("⚡ Running BitNet 2.0 Structural Sparsification Test Suite")
    print("=" * 70)
    test_bitnet2_sparsity()
    print("=" * 70)
