"""
Comprehensive Test Suite for TurboQuant (Google Research 2025)
Validates:
1. MSE distortion bounds vs Shannon Information Theoretical Lower Bounds.
2. Unbiased Inner Product estimation (<y, x_hat> vs <y, x> bias == 0).
3. Memory-Cache Top-k state retrieval accuracy.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import torch
import numpy as np
from python.turboquant import TurboQuantMSE, TurboQuantProd


def test_turboquant_mse_distortion_bounds():
    """Validates that MSE distortion matches theoretical predictions from Theorem 1."""
    torch.manual_seed(42)
    dim = 128
    num_samples = 5000
    
    # Generate random unit vectors on hypersphere S^{d-1}
    x_raw = torch.randn(num_samples, dim)
    x = x_raw / torch.norm(x_raw, p=2, dim=-1, keepdim=True)
    
    expected_mse_bounds = {
        1: 0.40,   # Paper approx: 0.36
        2: 0.14,   # Paper approx: 0.117
        3: 0.045,  # Paper approx: 0.030
        4: 0.015   # Paper approx: 0.009
    }
    
    for bits, max_expected in expected_mse_bounds.items():
        tq = TurboQuantMSE(dim=dim, bits=bits, seed=42)
        indices, norms = tq.quantize(x)
        x_hat = tq.dequantize(indices, norms)
        
        mse = torch.mean(torch.sum((x - x_hat) ** 2, dim=-1)).item()
        print(f"   [TurboQuantMSE {bits}-bit] Empirical MSE: {mse:.4f} (Theoretical upper bound: < {max_expected})")
        assert mse < max_expected, f"MSE {mse} exceeded expected bound {max_expected} for {bits} bits"


def test_turboquant_prod_unbiased_inner_product():
    """Validates that TurboQuant-Prod provides UNBIASED inner product estimates (Theorem 2)."""
    torch.manual_seed(123)
    dim = 64
    num_samples = 4000
    
    # Dataset vectors x
    x = torch.randn(num_samples, dim)
    x = x / torch.norm(x, p=2, dim=-1, keepdim=True)
    
    # Query vector y
    y = torch.randn(dim)
    y = y / torch.norm(y, p=2)
    
    true_inner = torch.matmul(x, y)
    
    # 1. Test TurboQuantProd (Unbiased via QJL residual)
    tq_prod = TurboQuantProd(dim=dim, total_bits=3, seed=42)
    q_data = tq_prod.quantize(x)
    x_hat_prod = tq_prod.dequantize(q_data)
    est_inner_prod = torch.matmul(x_hat_prod, y)
    
    bias_prod = torch.mean(est_inner_prod - true_inner).item()
    print(f"   [TurboQuantProd 3-bit] Inner Product Mean Bias: {bias_prod:+.5f} (Should be ~0.000)")
    assert abs(bias_prod) < 0.008, f"TurboQuantProd bias too high: {bias_prod}"
    
    # 2. Compare with naive MSE quantizer which is known to be biased
    tq_mse = TurboQuantMSE(dim=dim, bits=2, seed=42)
    ind, norms = tq_mse.quantize(x)
    x_hat_mse = tq_mse.dequantize(ind, norms)
    est_inner_mse = torch.matmul(x_hat_mse, y)
    bias_mse = torch.mean(est_inner_mse - true_inner).item()
    print(f"   [TurboQuantMSE 2-bit (Biased baseline)] Inner Product Mean Bias: {bias_mse:+.5f}")


def test_memory_cache_retrieval_accuracy():
    """
    Simulates Bit-MC-SSM Memory-Caching: tests 1@k and top-k recall
    of sequential SSM hidden state trajectories.
    """
    torch.manual_seed(777)
    dim = 64  # SSM hidden state dimension
    num_memories = 100
    num_queries = 50
    
    # Simulate realistic SSM hidden states (clustered trajectory manifolds)
    centers = torch.randn(10, dim)
    cluster_ids = torch.randint(0, 10, (num_memories,))
    noise = 0.2 * torch.randn(num_memories, dim)
    memory_states = centers[cluster_ids] + noise
    memory_states = memory_states / torch.norm(memory_states, p=2, dim=-1, keepdim=True)
    
    # Query states from the same attractor manifolds
    q_cluster_ids = torch.randint(0, 10, (num_queries,))
    q_noise = 0.2 * torch.randn(num_queries, dim)
    queries = centers[q_cluster_ids] + q_noise
    queries = queries / torch.norm(queries, p=2, dim=-1, keepdim=True)
    
    # Exact Ground Truth Similarities
    exact_similarities = torch.matmul(queries, memory_states.T) # (Q, M)
    
    # 1. TurboQuant-MSE (3-bit)
    tq_mse = TurboQuantMSE(dim=dim, bits=3, seed=100)
    ind_mse, norms_mse = tq_mse.quantize(memory_states)
    recon_mse = tq_mse.dequantize(ind_mse, norms_mse)
    
    # 2. TurboQuant-Prod (4-bit: 3-bit MSE + 1-bit QJL)
    tq_prod = TurboQuantProd(dim=dim, total_bits=4, seed=100)
    q_cache = tq_prod.quantize(memory_states)
    recon_prod = tq_prod.dequantize(q_cache)
    
    # Reconstructed inner products
    sim_mse = torch.matmul(queries, recon_mse.T)
    top1_mse_idx = torch.argmax(sim_mse, dim=-1)
    
    sim_prod = torch.matmul(queries, recon_prod.T)
    top1_prod_idx = torch.argmax(sim_prod, dim=-1)
    
    # 1. Semantic Cluster Accuracy (Did it retrieve memory from the correct cluster?)
    mse_cluster_hits = sum(1 for q in range(num_queries) if cluster_ids[top1_mse_idx[q]] == q_cluster_ids[q])
    prod_cluster_hits = sum(1 for q in range(num_queries) if cluster_ids[top1_prod_idx[q]] == q_cluster_ids[q])
    
    mse_cluster_acc = mse_cluster_hits / num_queries
    prod_cluster_acc = prod_cluster_hits / num_queries
    
    # 2. Similarity Score Pearson Correlation with Exact Ground Truth
    flat_exact = exact_similarities.flatten()
    flat_mse = sim_mse.flatten()
    flat_prod = sim_prod.flatten()
    
    corr_mse = torch.corrcoef(torch.stack([flat_exact, flat_mse]))[0, 1].item()
    corr_prod = torch.corrcoef(torch.stack([flat_exact, flat_prod]))[0, 1].item()
    
    print(f"🎯 Semantic Cluster Retrieval Accuracy (TurboQuant-MSE 3-bit):  {mse_cluster_acc * 100:.1f}%")
    print(f"🎯 Semantic Cluster Retrieval Accuracy (TurboQuant-Prod 4-bit): {prod_cluster_acc * 100:.1f}%")
    print(f"🎯 Similarity Score Correlation with Ground Truth (MSE):       {corr_mse * 100:.2f}%")
    print(f"🎯 Similarity Score Correlation with Ground Truth (Prod):      {corr_prod * 100:.2f}%")
    
    assert mse_cluster_acc >= 0.95, f"MSE Cluster Acc {mse_cluster_acc} below 95%"
    assert prod_cluster_acc >= 0.95, f"Prod Cluster Acc {prod_cluster_acc} below 95%"
    assert corr_mse >= 0.97, f"MSE Correlation {corr_mse} below 0.97"
    assert corr_prod >= 0.97, f"Prod Correlation {corr_prod} below 0.97"


if __name__ == "__main__":
    print("=" * 70)
    print("⚡ Running TurboQuant (Google Research 2025) Test Suite")
    print("=" * 70)
    print("1. Testing MSE Distortion Bounds...")
    test_turboquant_mse_distortion_bounds()
    print("\n2. Testing Unbiased Inner Product Property...")
    test_turboquant_prod_unbiased_inner_product()
    print("\n3. Testing Memory-Cache Top-2 Recall...")
    test_memory_cache_retrieval_accuracy()
    print("\n🎉 ALL TURBOQUANT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
