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


class EncoderStage(nn.Module):
    """A hierarchical stage containing downsampling and Spatial Bi-Delta-SSM blocks."""
    def __init__(self, in_chans: int, out_chans: int, grid_size: tuple, downsample: bool = True, num_blocks: int = 1, d_state: int = 32, tau: float = 0.85):
        super().__init__()
        self.grid_size = grid_size
        if downsample:
            self.down = nn.Conv2d(in_chans, out_chans, kernel_size=2, stride=2)
        else:
            self.down = nn.Identity()
            
        num_patches = grid_size[0] * grid_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, out_chans))
        
        self.blocks = nn.ModuleList([
            SpatialBiBitMCSSMBlock(d_model=out_chans, grid_size=grid_size, d_state=d_state, tau=tau)
            for _ in range(num_blocks)
        ])
        self.norm = FusedRMSNorm(out_chans)
        
    def forward(self, x: torch.Tensor):
        x = self.down(x)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_flat = x_flat + self.pos_embed
        
        for block in self.blocks:
            x_flat = block(x_flat)
            
        x_flat = self.norm(x_flat)
        return x_flat.transpose(1, 2).view(B, C, H, W)


class UNetDecoder(nn.Module):
    """
    U-Net Decoder with Skip Connections.
    Takes multi-scale features from the hierarchical encoder and fuses them.
    """
    def __init__(self, dims: list):
        super().__init__()
        # dims = [C, 2C, 4C, 8C]
        self.up1 = nn.ConvTranspose2d(dims[3], dims[2], kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dims[2] * 2, dims[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(dims[2]),
            nn.SiLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(dims[2], dims[1], kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(dims[1] * 2, dims[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.SiLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(dims[1], dims[0], kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(dims[0] * 2, dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.SiLU(inplace=True)
        )
        
        # Stage 1 has H/4 resolution, so upsample by 4 to get original H
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(dims[0], 32, kernel_size=4, stride=4),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, features):
        f1, f2, f3, f4 = features
        
        x = self.up1(f4)
        x = torch.cat([x, f3], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, f2], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x, f1], dim=1)
        x = self.conv3(x)
        
        x = self.final_up(x)
        return x


class BitSegNet(nn.Module):
    """
    Bit-SegNet v3: Hierarchical U-Net with 1.58-bit Bi-Delta-SSM core.
    Provides pixel-perfect boundaries by fusing high-res skip connections
    with global SSM context.
    """
    def __init__(self, img_size=256, in_chans=3, base_dim=64,
                 depths=[1, 1, 2, 1], d_state=32, tau=0.85):
        super().__init__()
        dims = [base_dim, base_dim*2, base_dim*4, base_dim*8]
        
        # Stage 1: H/4
        self.patch_embed = nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4)
        self.stage1 = EncoderStage(dims[0], dims[0], (img_size // 4, img_size // 4), 
                                   downsample=False, num_blocks=depths[0], d_state=d_state, tau=tau)
        
        # Stage 2: H/8
        self.stage2 = EncoderStage(dims[0], dims[1], (img_size // 8, img_size // 8), 
                                   downsample=True, num_blocks=depths[1], d_state=d_state, tau=tau)
        
        # Stage 3: H/16
        self.stage3 = EncoderStage(dims[1], dims[2], (img_size // 16, img_size // 16), 
                                   downsample=True, num_blocks=depths[2], d_state=d_state, tau=tau)
        
        # Stage 4: H/32
        self.stage4 = EncoderStage(dims[2], dims[3], (img_size // 32, img_size // 32), 
                                   downsample=True, num_blocks=depths[3], d_state=d_state, tau=tau)
        
        self.decoder = UNetDecoder(dims)

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        
        return self.decoder([f1, f2, f3, f4])


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
    print("🚀 Initializing U-Bit-SegNet (v3: Hierarchical U-Net 1.58-bit SSM)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tiny configuration for quick testing (img_size must be div by 32)
    model = BitSegNet(img_size=64, base_dim=32, depths=[1, 1, 1, 1]).to(device)

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

    print(f"💎 Decoder is now a Hierarchical U-Net with Skip Connections!")
