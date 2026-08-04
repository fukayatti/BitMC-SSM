"""
Test Suite for GaLore Optimizer (Memory Reduction & Convergence)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from python.galore_optimizer import GaLoreAdamW


def test_galore_optimizer_memory_and_convergence():
    torch.manual_seed(42)
    
    d_in = 256
    d_out = 256
    rank = 16
    
    model = nn.Linear(d_in, d_out, bias=False)
    
    # Create low-rank manifold target (rank 16)
    target_A = torch.randn(d_out, rank)
    target_B = torch.randn(rank, d_in)
    target_W = torch.matmul(target_A, target_B)
    
    optimizer = GaLoreAdamW(model.parameters(), lr=0.05, rank=rank, update_proj_gap=20)
    
    # Verify optimizer state memory
    x = torch.randn(64, d_in)
    target_y = torch.matmul(x, target_W.T)
    
    # 1 step to initialize state
    output = model(x)
    loss = nn.functional.mse_loss(output, target_y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Check allocated optimizer state dimensions
    state = optimizer.state[model.weight]
    assert state['is_galore'] is True
    exp_avg_shape = state['exp_avg'].shape
    print(f"   [GaLore State] Full weight shape: {model.weight.shape} (65,536 elements)")
    print(f"   [GaLore State] Low-Rank Adam moment shape: {exp_avg_shape} ({exp_avg_shape[0] * exp_avg_shape[1]} elements)")
    
    memory_savings = (1.0 - (exp_avg_shape[0] * exp_avg_shape[1]) / (d_in * d_out)) * 100
    print(f"   🚀 Optimizer Memory Reduction: {memory_savings:.1f}% saved!")
    assert memory_savings >= 90.0, "Expected >= 90% memory savings"
    
    # Test convergence over 250 steps
    initial_loss = loss.item()
    for step in range(250):
        output = model(x)
        loss = nn.functional.mse_loss(output, target_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    final_loss = loss.item()
    print(f"   📉 Training Loss: {initial_loss:.4f} -> {final_loss:.4f} (Reduction: {(1.0 - final_loss/initial_loss)*100:.1f}%)")
    assert final_loss < initial_loss * 0.1, "GaLore failed to optimize loss"
    print("   ✅ GaLore convergence & memory test PASSED!")


if __name__ == "__main__":
    print("=" * 70)
    print("⚡ Running GaLore (Gradient Low-Rank Projection) Test Suite")
    print("=" * 70)
    test_galore_optimizer_memory_and_convergence()
    print("=" * 70)
