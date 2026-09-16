"""
Bit-SegNet: 1.58-bit Bi-Delta-SSM Background Removal / Segmentation Network
============================================================================
Reuses the same H-BitLinear (ternary, zero-GEMM) + Bi-Delta-SSM building
blocks as bit_clip.py, adapted for dense per-pixel mask prediction instead
of a single pooled embedding vector.

Two changes vs. the CLIP encoder (bit_clip.py's BitImageEncoder):

1. Spatial (2D-aware) scanning: a naive 1D raster-scan bidirectional SSM
   (row-major flatten) loses vertical adjacency -- the patch directly above
   another patch ends up far away in the flattened sequence. This adds a
   second bidirectional scan along the transposed (column-major) order and
   combines both, similar in spirit to the multi-direction scans used in
   Vision Mamba / VMamba.

2. Lightweight segmentation head: instead of a CNN decoder with transposed
   convolutions, each patch's feature vector is projected directly to a
   (patch_size x patch_size) block of mask logits and reassembled into the
   full-resolution mask. No upsampling layers, no skip connections yet --
   the simplest thing that can produce a same-resolution mask; skip
   connections can be added later if boundary sharpness needs it.

Usage: python python/bit_segnet.py  (runs a self-test forward/backward pass)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from h_bitlinear import HBitLinear
from train import FusedRMSNorm
from bit_clip import BiDeltaSSMBlock


class SpatialBiDeltaSSMBlock(nn.Module):
    """
    Runs a bidirectional Delta-SSM scan along rows (horizontal) AND a second
    one along columns (vertical, via transposing the patch grid before/after
    the scan), then fuses both. This preserves vertical spatial adjacency
    that a single row-major bidirectional scan cannot see.
    """
    def __init__(self, d_model: int, grid_size: tuple, d_state: int = 32, tau: float = 0.85):
        super().__init__()
        self.grid_h, self.grid_w = grid_size
        self.row_ssm = BiDeltaSSMBlock(d_model=d_model, d_state=d_state, tau=tau)
        self.col_ssm = BiDeltaSSMBlock(d_model=d_model, d_state=d_state, tau=tau)
        self.combine = HBitLinear(d_model * 2, d_model, tau=tau, use_hadamard=True)

    def forward(self, x: torch.Tensor):
        # x: (B, H*W, D) in row-major patch order
        B, N, D = x.shape
        H, W = self.grid_h, self.grid_w

        row_out = self.row_ssm(x)

        x_grid = x.view(B, H, W, D)
        x_col_major = x_grid.transpose(1, 2).reshape(B, W * H, D)
        col_out = self.col_ssm(x_col_major)
        col_out = col_out.view(B, W, H, D).transpose(1, 2).reshape(B, H * W, D)

        return self.combine(torch.cat([row_out, col_out], dim=-1))


class SpatialBiBitMCSSMBlock(nn.Module):
    """Same pre-norm SSM+SwiGLU-FFN residual structure as bit_clip.py's
    BiBitMCSSMBlock, but with the spatially-aware SSM above in place of a
    plain row-major bidirectional scan."""
    def __init__(self, d_model: int, grid_size: tuple, d_state: int = 32, tau: float = 0.85):
        super().__init__()
        self.norm1 = FusedRMSNorm(d_model)
        self.ssm = SpatialBiDeltaSSMBlock(d_model=d_model, grid_size=grid_size, d_state=d_state, tau=tau)
        self.norm2 = FusedRMSNorm(d_model)
        self.ffn_in = HBitLinear(d_model, d_model * 4, tau=tau, use_hadamard=False)
        self.ffn_out = HBitLinear(d_model * 2, d_model, tau=tau, use_hadamard=True)

    def forward(self, x: torch.Tensor):
        x = x + self.ssm(self.norm1(x))
        ffn_p = self.ffn_in(self.norm2(x))
        f1, f2 = ffn_p.chunk(2, dim=-1)
        gated = F.silu(f1) * f2
        x = x + self.ffn_out(gated)
        return x


class SegmentationHead(nn.Module):
    """Projects each patch's feature vector directly to a
    (patch_size x patch_size) block of mask logits -- no deconvolution /
    upsampling layers, just a linear projection and a reshape back to full
    image resolution."""
    def __init__(self, d_model: int, patch_size: int, tau: float = 0.85):
        super().__init__()
        self.patch_size = patch_size
        self.proj = HBitLinear(d_model, patch_size * patch_size, tau=tau, use_hadamard=False)

    def forward(self, x: torch.Tensor, grid_size: tuple):
        H, W = grid_size
        B, N, D = x.shape
        P = self.patch_size
        mask_patches = self.proj(x)                       # (B, N, P*P)
        mask_patches = mask_patches.view(B, H, W, P, P)
        mask_patches = mask_patches.permute(0, 1, 3, 2, 4)  # (B, H, P, W, P)
        return mask_patches.reshape(B, 1, H * P, W * P)     # (B, 1, img_h, img_w) raw logits


class BitSegNet(nn.Module):
    """
    1.58-bit Bi-Delta-SSM background removal network.
    Input:  (B, 3, img_size, img_size) RGB image
    Output: (B, 1, img_size, img_size) raw foreground-mask logits
            (apply sigmoid for a [0,1] alpha/probability mask)
    """
    def __init__(self, img_size=256, patch_size=16, in_chans=3, d_model=384,
                 n_layers=6, d_state=32, tau=0.85):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        num_patches = self.grid_size[0] * self.grid_size[1]

        self.patch_embed = nn.Conv2d(in_chans, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))

        self.blocks = nn.ModuleList([
            SpatialBiBitMCSSMBlock(d_model=d_model, grid_size=self.grid_size, d_state=d_state, tau=tau)
            for _ in range(n_layers)
        ])
        self.norm_f = FusedRMSNorm(d_model)
        self.seg_head = SegmentationHead(d_model, patch_size, tau=tau)

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)               # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)      # (B, N, D)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm_f(x)
        return self.seg_head(x, self.grid_size)  # (B, 1, img_size, img_size) logits


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss -- complements pixel-wise BCE with a region-overlap
    signal, standard for segmentation/matting since BCE alone is easily
    dominated by the (usually much larger) background class."""
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    return bce + dice


if __name__ == "__main__":
    print("🚀 Initializing Bit-SegNet (1.58-bit Bi-Delta-SSM Background Removal Network)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tiny configuration for quick testing
    model = BitSegNet(img_size=64, patch_size=16, d_model=128, n_layers=2).to(device)

    B = 2
    dummy_images = torch.randn(B, 3, 64, 64, device=device)
    dummy_masks = (torch.rand(B, 1, 64, 64, device=device) > 0.5).float()

    logits = model(dummy_images)
    print(f"\n✅ Forward Pass Successful! Output shape: {logits.shape} (expected {dummy_masks.shape})")
    assert logits.shape == dummy_masks.shape

    loss = segmentation_loss(logits, dummy_masks)
    print(f"📉 Initial Segmentation Loss (BCE + Dice): {loss.item():.4f}")

    loss.backward()
    print("✅ Backward Pass Successful! (Gradients computed successfully)")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model Parameters: {total_params / 1e6:.2f}M ({total_params:,} params)")

    seg_head_weights = model.seg_head.proj.weight
    gamma = seg_head_weights.abs().mean().clamp(min=1e-5)
    w_scaled = seg_head_weights / gamma
    zero_ratio = (w_scaled.abs() <= 0.85).float().mean().item() * 100.0
    print(f"💎 Segmentation Head 1.58-bit Weight Zero-Sparsity: {zero_ratio:.1f}%")
