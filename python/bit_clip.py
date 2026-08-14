import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from h_bitlinear import HBitLinear
# Import dependencies from train.py that contain Triton optimizations and 1.58-bit DeltaSSM
from train import DeltaSSMBlock, FusedRMSNorm, fused_silu_gating

class BiDeltaSSMBlock(nn.Module):
    """
    Bidirectional 1.58-bit Delta-SSM.
    Scans the sequence forward and backward simultaneously, then combines the outputs.
    Perfect for Encoder-only tasks (Vision, Text embeddings) replacing Bidirectional Transformers.
    """
    def __init__(self, d_model: int, d_state: int = 32, tau: float = 0.85):
        super().__init__()
        self.forward_ssm = DeltaSSMBlock(d_model, d_state, tau)
        self.backward_ssm = DeltaSSMBlock(d_model, d_state, tau)
        # Project the concatenated forward+backward features back to d_model
        self.out_proj = HBitLinear(d_model * 2, d_model, tau=tau, use_hadamard=True)
        
    def forward(self, x: torch.Tensor):
        # 1. Forward Scan
        y_fwd, _ = self.forward_ssm(x)
        
        # 2. Backward Scan
        x_rev = x.flip(1)
        y_rev, _ = self.backward_ssm(x_rev)
        y_rev = y_rev.flip(1)
        
        # 3. Combine
        y_cat = torch.cat([y_fwd, y_rev], dim=-1)
        return self.out_proj(y_cat)

class BiBitMCSSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 32, tau: float = 0.85):
        super().__init__()
        self.norm1 = FusedRMSNorm(d_model)
        self.ssm = BiDeltaSSMBlock(d_model=d_model, d_state=d_state, tau=tau)
        self.norm2 = FusedRMSNorm(d_model)
        self.ffn_in = HBitLinear(d_model, d_model * 4, tau=tau, use_hadamard=False)
        self.ffn_out = HBitLinear(d_model * 2, d_model, tau=tau, use_hadamard=True)

    def forward(self, x: torch.Tensor):
        ssm_out = self.ssm(self.norm1(x))
        x = x + ssm_out
        ffn_p = self.ffn_in(self.norm2(x))
        f1, f2 = ffn_p.chunk(2, dim=-1)
        if fused_silu_gating is not None and x.is_cuda:
            gated = fused_silu_gating(f1, f2)
        else:
            gated = F.silu(f1) * f2
        x = x + self.ffn_out(gated)
        return x

class BitImageEncoder(nn.Module):
    """
    1.58-bit Vision Encoder (Vision Mamba style).
    Treats 16x16 image patches as a sequence and processes them with Bi-Delta-SSM.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, d_model=384, n_layers=6, d_state=32, tau=0.85):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, d_model, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        
        self.blocks = nn.ModuleList([
            BiBitMCSSMBlock(d_model=d_model, d_state=d_state, tau=tau)
            for _ in range(n_layers)
        ])
        self.norm_f = FusedRMSNorm(d_model)

    def forward(self, x: torch.Tensor):
        # x: (B, C, H, W)
        x = self.patch_embed(x)  # (B, d_model, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, d_model)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm_f(x)
        # Mean pooling to get a single vector representing the entire image
        return x.mean(dim=1)

class BitTextEncoder(nn.Module):
    """
    1.58-bit Text Encoder (BERT style).
    """
    def __init__(self, vocab_size=50257, d_model=384, n_layers=6, d_state=32, tau=0.85):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            BiBitMCSSMBlock(d_model=d_model, d_state=d_state, tau=tau)
            for _ in range(n_layers)
        ])
        self.norm_f = FusedRMSNorm(d_model)

    def forward(self, idx: torch.Tensor):
        # idx: (B, L)
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        # Mean pooling to get a single vector representing the entire text
        return x.mean(dim=1)

class BitCLIP(nn.Module):
    """
    Bit-CLIP: Contrastive Language-Image Pretraining with 1.58-bit Bi-Delta-SSMs.
    """
    def __init__(self, 
                 embed_dim=512,
                 vocab_size=50257, 
                 img_size=224, 
                 patch_size=16,
                 d_model=384, 
                 n_layers=6, 
                 d_state=32, 
                 tau=0.85):
        super().__init__()
        self.image_encoder = BitImageEncoder(img_size, patch_size, 3, d_model, n_layers, d_state, tau)
        self.text_encoder = BitTextEncoder(vocab_size, d_model, n_layers, d_state, tau)
        
        # Projection to joint multimodal space
        self.image_proj = nn.Linear(d_model, embed_dim, bias=False)
        self.text_proj = nn.Linear(d_model, embed_dim, bias=False)
        
        # Learnable temperature parameter (initialized to log(1/0.07) as in standard CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, image: torch.Tensor, text: torch.Tensor):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(text)
        
        image_embeds = self.image_proj(image_features)
        text_embeds = self.text_proj(text_features)
        
        # L2 Normalize embeddings to lay them out on a hypersphere
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        # Scaled pairwise dot products (B x B matrix)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text

def bit_clip_loss(logits_per_image, logits_per_text):
    """
    Softmax InfoNCE Loss for Bit-CLIP.
    Forces the diagonal elements (correct pairs) to approach 1.0, and scatters everything else.
    Due to BitMC-SSM's tiny memory footprint, this thrives on MASSIVE batch sizes (e.g. 4096).
    """
    B = logits_per_image.shape[0]
    labels = torch.arange(B, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2.0


if __name__ == "__main__":
    print("🚀 Initializing Bit-CLIP (1.58-bit Bi-Delta-SSM Vision-Language Model)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tiny configuration for quick testing
    model = BitCLIP(embed_dim=256, d_model=128, n_layers=2).to(device)
    
    # Dummy Data: Batch size 4
    B = 4
    dummy_images = torch.randn(B, 3, 224, 224, device=device) # 4 images
    dummy_texts = torch.randint(0, 50257, (B, 32), device=device) # 4 text sequences (length 32)
    
    # Forward Pass
    logits_per_image, logits_per_text = model(dummy_images, dummy_texts)
    
    print("\n✅ Forward Pass Successful!")
    print(f"   Logits Matrix Shape: {logits_per_image.shape} (Batch x Batch)")
    
    # Loss Calculation
    loss = bit_clip_loss(logits_per_image, logits_per_text)
    print(f"\n📉 Initial Softmax InfoNCE Loss: {loss.item():.4f}")
    
    # Backward Pass (Testing Gradients)
    loss.backward()
    print("✅ Backward Pass Successful! (Gradients computed successfully)")
    
    # Sparsity / 1.58-bit Check
    text_proj_weights = model.text_encoder.blocks[0].ffn_in.weight
    gamma = text_proj_weights.abs().mean().clamp(min=1e-5)
    w_scaled = text_proj_weights / gamma
    zero_ratio = (w_scaled.abs() <= 0.85).float().mean().item() * 100.0
    print(f"💎 Model 1.58-bit Weight Zero-Sparsity: {zero_ratio:.1f}%")
