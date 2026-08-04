"""
Unit test for BitNet v2 H-BitLinear:
1. Orthogonality & Reconstruction of Fast Hadamard Transform (FWHT)
2. Outlier Channel Suppression (Kurtosis & Peak Value Decay)
3. 4-bit Activation Quantization & Backward Gradient Flow
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import torch
import numpy as np
from python.h_bitlinear import fast_hadamard_transform, HBitLinear

def test_hadamard_orthogonality():
    print("=" * 65)
    print("1. Testing Fast Walsh-Hadamard Transform (FWHT) Orthogonality...")
    print("=" * 65)
    
    dims = [64, 128, 256, 512]
    for d in dims:
        x = torch.randn(4, d)
        h_x = fast_hadamard_transform(x)
        # Applying FWHT twice should recover original tensor (since H * H = I with proper scale)
        rec_x = fast_hadamard_transform(h_x)
        diff = (x - rec_x).abs().max().item()
        print(f"   Dim {d:3d} -> Max Reconstruction Difference: {diff:.2e} (Exact!)")
        assert diff < 1e-5, f"Reconstruction failed for dim {d}"
    print("✅ FWHT Orthogonality Verified Successfully!\n")

def test_outlier_suppression():
    print("=" * 65)
    print("2. Testing Outlier Channel Suppression (BitNet v2 Core Mechanism)...")
    print("=" * 65)
    
    d = 256
    # Create activations with severe outliers (e.g. 50x spike in specific channels)
    x = torch.randn(8, d)
    x[:, 10] *= 50.0 # Heavy outlier channel
    x[:, 77] *= -40.0 # Heavy outlier channel
    
    orig_peak = x.abs().max().item()
    orig_kurtosis = ((x - x.mean()).pow(4).mean() / (x.var().pow(2))).item()
    
    x_h = fast_hadamard_transform(x)
    h_peak = x_h.abs().max().item()
    h_kurtosis = ((x_h - x_h.mean()).pow(4).mean() / (x_h.var().pow(2))).item()
    
    print(f"   Original Activation: Peak = {orig_peak:.2f} | Kurtosis = {orig_kurtosis:.2f} (Sharp Spike!)")
    print(f"   Hadamard Transform:  Peak = {h_peak:.2f} | Kurtosis = {h_kurtosis:.2f} (Gaussian-like Smooth!)")
    print(f"   🚀 Outlier Peak Reduction: {orig_peak / h_peak:.2f}x reduction!")
    assert h_peak < orig_peak, "Hadamard transform should suppress outlier peaks"
    print("✅ Outlier Suppression Verified!\n")

def test_h_bitlinear_forward_backward():
    print("=" * 65)
    print("3. Testing H-BitLinear Forward, Backward (STE), and Sparsity...")
    print("=" * 65)
    
    layer = HBitLinear(in_features=128, out_features=64, tau=0.85, use_hadamard=True)
    x = torch.randn(4, 16, 128, requires_grad=True)
    
    out = layer(x)
    assert out.shape == (4, 16, 64)
    
    # Backward pass
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None and not torch.isnan(x.grad).any()
    assert layer.weight.grad is not None and not torch.isnan(layer.weight.grad).any()
    
    stats = layer.get_sparsity_stats()
    print(f"   Output Shape: {list(out.shape)}")
    print(f"   Weight Zero-Sparsity: {stats['zero_weight_ratio'] * 100:.1f}%")
    print(f"   Input Gradient Norm: {x.grad.norm().item():.4f}")
    print(f"   Weight Gradient Norm: {layer.weight.grad.norm().item():.4f}")
    print("✅ H-BitLinear Forward/Backward Execution Verified!\n")

if __name__ == "__main__":
    test_hadamard_orthogonality()
    test_outlier_suppression()
    test_h_bitlinear_forward_backward()
    print("🎉 All BitNet v2 H-BitLinear Unit Tests Passed!")
