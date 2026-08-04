"""
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
Reference: Google Research & NYU (arXiv:2504.19874v1, 2025)
Authors: Amir Zandieh, Majid Daliri, Majid H., Vahab Mirrokni

Features:
- TurboQuantMSE: Near-optimal MSE scalar quantization with random orthogonal rotation.
- TurboQuantProd: Unbiased inner-product vector quantization with 1-bit QJL on residual.
"""

import math
import torch
import torch.nn as nn
import numpy as np


# Standard Lloyd-Max optimal centroids for N(0, 1)
LLOYD_MAX_CENTROIDS = {
    1: [-0.79788456, 0.79788456],
    2: [-1.510418, -0.452784, 0.452784, 1.510418],
    3: [-2.15224, -1.34393, -0.75601, -0.24508, 0.24508, 0.75601, 1.34393, 2.15224],
    4: [-2.7326, -2.0694, -1.6181, -1.2562, -0.9424, -0.6568, -0.3881, -0.1284,
         0.1284,  0.3881,  0.6568,  0.9424,  1.2562,  1.6181,  2.0694,  2.7326],
}


def generate_orthogonal_matrix(dim: int, seed: int = 42) -> torch.Tensor:
    """Generates a random orthogonal matrix Pi via QR decomposition."""
    gen = torch.Generator().manual_seed(seed)
    h = torch.randn(dim, dim, generator=gen)
    q, r = torch.linalg.qr(h)
    d = torch.diagonal(r, 0).sign()
    q = q * d.unsqueeze(0)
    return q


def generate_gaussian_matrix(dim: int, seed: int = 1337) -> torch.Tensor:
    """Generates a random Gaussian projection matrix for QJL."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(dim, dim, generator=gen)


class TurboQuantMSE:
    """
    TurboQuant-MSE: Data-oblivious vector quantizer minimizing MSE distortion.
    Achieves MSE within factor of 2.7 of Shannon's theoretical lower bound.
    """
    def __init__(self, dim: int, bits: int = 2, seed: int = 42, device: str = "cpu"):
        assert bits in LLOYD_MAX_CENTROIDS, f"Bits must be in {list(LLOYD_MAX_CENTROIDS.keys())}"
        self.dim = dim
        self.bits = bits
        self.device = device
        
        # Orthogonal rotation matrix Pi (d x d)
        pi = generate_orthogonal_matrix(dim, seed=seed).to(device)
        self.register_buffer = pi
        self.pi = pi
        
        # Scaled centroids: N(0, 1/d) standard deviation = 1 / sqrt(d)
        centroids_base = torch.tensor(LLOYD_MAX_CENTROIDS[bits], dtype=torch.float32, device=device)
        self.centroids = centroids_base / math.sqrt(dim)
        
        # Decision boundaries for fast bucket assignment (midpoints)
        self.boundaries = (self.centroids[:-1] + self.centroids[1:]) / 2.0

    def quantize(self, x: torch.Tensor):
        """
        Quantizes vector x of shape (..., d).
        Returns:
            indices: torch.Tensor of shape (..., d) with integer codes in [0, 2^b - 1]
            norms: torch.Tensor of shape (..., 1) with original L2 norms
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim).to(self.device, dtype=torch.float32)
        
        # Compute L2 norms
        norms = torch.norm(x_flat, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        x_norm = x_flat / norms
        
        # Random orthogonal rotation: y = x_norm @ Pi
        y = torch.matmul(x_norm, self.pi)
        
        # Scalar quantization: bucketize into centroids
        # bucketize returns index in [0, len(boundaries)]
        indices = torch.bucketize(y, self.boundaries)
        
        indices = indices.reshape(orig_shape)
        norms = norms.reshape(orig_shape[:-1] + (1,))
        return indices, norms

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        """
        Dequantizes indices and scales by original norms.
        Returns reconstructed vector of shape (..., d).
        """
        orig_shape = indices.shape
        ind_flat = indices.reshape(-1, self.dim).to(self.device)
        norms_flat = norms.reshape(-1, 1).to(self.device)
        
        # Lookup centroids
        y_hat = self.centroids[ind_flat]
        
        # Inverse rotation: x_hat = y_hat @ Pi^T
        x_norm_hat = torch.matmul(y_hat, self.pi.T)
        
        # Rescale by norm
        x_hat = x_norm_hat * norms_flat
        return x_hat.reshape(orig_shape)


class TurboQuantProd:
    """
    TurboQuant-Prod: Unbiased Inner-Product Vector Quantizer.
    Combines (b - 1)-bit MSE TurboQuant with 1-bit QJL (Quantized Johnson-Lindenstrauss)
    on the residual vector, guaranteeing zero-bias inner product estimation.
    """
    def __init__(self, dim: int, total_bits: int = 3, seed: int = 42, device: str = "cpu"):
        assert total_bits >= 2, "Total bits for TurboQuantProd must be >= 2 (at least 1-bit MSE + 1-bit QJL)"
        self.dim = dim
        self.total_bits = total_bits
        self.mse_bits = total_bits - 1
        self.device = device
        
        # Stage 1: MSE quantizer
        self.mse_quantizer = TurboQuantMSE(dim, bits=self.mse_bits, seed=seed, device=device)
        
        # Stage 2: QJL Gaussian matrix S (d x d)
        self.s_matrix = generate_gaussian_matrix(dim, seed=seed + 100).to(device)
        
        # QJL scaling constant: sqrt(pi / (2 * d^2))
        self.qjl_scale = math.sqrt(math.pi / (2.0 * (dim ** 2)))

    def quantize(self, x: torch.Tensor):
        """
        Quantizes vector x of shape (..., d).
        Returns:
            mse_indices: (..., d) tensor of integer codes
            norms: (..., 1) tensor of L2 norms of x
            qjl_bits: (..., d) boolean tensor of 1-bit signs (stored as uint8/bool)
            gamma: (..., 1) tensor of residual L2 norms
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim).to(self.device, dtype=torch.float32)
        
        # Stage 1: MSE Quantization
        mse_indices, norms = self.mse_quantizer.quantize(x_flat)
        x_mse = self.mse_quantizer.dequantize(mse_indices, norms)
        
        # Compute residual: r = x - x_mse
        r = x_flat - x_mse
        gamma = torch.norm(r, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        r_normalized = r / gamma
        
        # Stage 2: 1-bit QJL projection: sign(r_norm @ S^T)
        s_proj = torch.matmul(r_normalized, self.s_matrix.T)
        qjl_signs = torch.sign(s_proj)
        qjl_signs[qjl_signs == 0] = 1.0  # Tie-breaker
        
        # Pack sign as boolean (True for +1, False for -1)
        qjl_bits = (qjl_signs > 0).to(torch.uint8)
        
        mse_indices = mse_indices.reshape(orig_shape)
        norms = norms.reshape(orig_shape[:-1] + (1,))
        qjl_bits = qjl_bits.reshape(orig_shape)
        gamma = gamma.reshape(orig_shape[:-1] + (1,))
        
        return {
            "mse_indices": mse_indices,
            "norms": norms,
            "qjl_bits": qjl_bits,
            "gamma": gamma
        }

    def dequantize(self, quantized_data: dict) -> torch.Tensor:
        """
        Dequantizes two-stage representation into reconstructed vector.
        """
        mse_indices = quantized_data["mse_indices"]
        norms = quantized_data["norms"]
        qjl_bits = quantized_data["qjl_bits"]
        gamma = quantized_data["gamma"]
        
        orig_shape = mse_indices.shape
        ind_flat = mse_indices.reshape(-1, self.dim)
        norms_flat = norms.reshape(-1, 1)
        bits_flat = qjl_bits.reshape(-1, self.dim)
        gamma_flat = gamma.reshape(-1, 1)
        
        # Stage 1 reconstruction
        x_mse = self.mse_quantizer.dequantize(ind_flat, norms_flat)
        
        # Stage 2 QJL reconstruction: gamma * sqrt(pi / (2 * d^2)) * (q @ S)
        q_signs = torch.where(bits_flat > 0, 1.0, -1.0).to(torch.float32)
        qjl_recon = torch.matmul(q_signs, self.s_matrix) * (self.qjl_scale * gamma_flat)
        
        x_hat = x_mse + qjl_recon
        return x_hat.reshape(orig_shape)

    def inner_product(self, query: torch.Tensor, quantized_data: dict) -> torch.Tensor:
        """
        Computes inner products directly against a query vector y: <y, x_hat>.
        Guaranteed to be an unbiased estimator of <y, x>.
        """
        x_hat = self.dequantize(quantized_data)
        return torch.sum(query * x_hat, dim=-1)
