import os
import argparse
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

# Import our custom models
from bit_segnet import BitSegNet, segmentation_loss
from galore_optimizer import GaLoreAdamW

class ImageMaskDataset(Dataset):
    """
    Generic Dataset for Background Removal / Segmentation.
    Assumes a directory structure like:
    data_dir/
      images/
        0001.jpg
        0002.png
      masks/
        0001.png
        0002.png
    Filenames must match (ignoring extensions).
    """
    def __init__(self, data_dir, img_size=256, is_train=True):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "images"
        self.mask_dir = self.data_dir / "masks"
        
        # Collect all image paths
        valid_exts = {'.jpg', '.jpeg', '.png'}
        self.image_paths = []
        if self.img_dir.exists():
            for f in self.img_dir.iterdir():
                if f.suffix.lower() in valid_exts:
                    self.image_paths.append(f)
        
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Try to find corresponding mask (assuming same stem, .png extension)
        mask_path = self.mask_dir / f"{img_path.stem}.png"
        
        if not mask_path.exists():
            # If not found, look for any extension
            found = False
            for ext in ['.jpg', '.jpeg', '.png']:
                alt_path = self.mask_dir / f"{img_path.stem}{ext}"
                if alt_path.exists():
                    mask_path = alt_path
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"Mask not found for image: {img_path.name} in {self.mask_dir}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale

        # 1. Resize both to target size
        image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.Resampling.NEAREST)

        # 2. Data Augmentation (Only for Training)
        if self.is_train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random Color Jitter (Apply ONLY to image)
            jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
            image = jitter(image)

        # 3. Convert to Tensor
        image_t = TF.to_tensor(image)
        mask_t = TF.to_tensor(mask)

        # 4. Normalize Image
        image_t = TF.normalize(image_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Binarize mask just in case
        mask_t = (mask_t > 0.5).float()

        return image_t, mask_t


def get_args_parser():
    parser = argparse.ArgumentParser('Bit-SegNet Background Removal Training', add_help=False)
    parser.add_argument('--data_dir', default='./data/seg', type=str,
                        help='dataset path containing "images" and "masks" folders')
    parser.add_argument('--output_dir', default='./checkpoints_seg', type=str,
                        help='path where to save checkpoints')
    
    # Model parameters
    parser.add_argument('--img_size', default=256, type=int, help='input image size')
    parser.add_argument('--base_dim', default=64, type=int, help='base channel dimension for stage 1')
    parser.add_argument('--depths', default="1,1,2,1", type=str, help='number of SSM blocks per stage (comma separated)')
    
    # Training parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--use_galore', action='store_true', help='Use GaLore optimizer for low memory footprint')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--resume', default='', type=str, help='path to checkpoint to resume from')
    
    return parser

def main():
    parser = argparse.ArgumentParser('Bit-SegNet training', parents=[get_args_parser()])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Dataset & DataLoader
    print(f"📁 Loading dataset from {args.data_dir}...")
    try:
        train_dir = os.path.join(args.data_dir, 'train')
        val_dir = os.path.join(args.data_dir, 'val')
        
        if os.path.exists(train_dir) and os.path.exists(val_dir):
            print("🔍 Found explicit 'train' and 'val' splits. Bypassing random_split.")
            train_dataset = ImageMaskDataset(train_dir, img_size=args.img_size, is_train=True)
            val_dataset = ImageMaskDataset(val_dir, img_size=args.img_size, is_train=False)
        else:
            print("🔍 No explicit 'train'/'val' folders. Performing automatic 90/10 random split.")
            full_dataset = ImageMaskDataset(args.data_dir, img_size=args.img_size, is_train=True)
            if len(full_dataset) == 0:
                raise FileNotFoundError("Dataset is empty.")
                
            val_size = max(1, int(0.1 * len(full_dataset)))
            train_size = len(full_dataset) - val_size
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size], generator=generator
            )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                num_workers=args.num_workers, pin_memory=True)
        print(f"✅ Dataset ready: {len(train_dataset)} Train, {len(val_dataset)} Val.")
    except FileNotFoundError:
        print(f"⚠️ Warning: Dataset not found at {args.data_dir}. Creating a DUMMY dataset for testing.")
        dummy = [(torch.randn(3, args.img_size, args.img_size), 
                  (torch.rand(1, args.img_size, args.img_size) > 0.5).float()) for _ in range(64)]
        train_loader = DataLoader(dummy[:50], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(dummy[50:], batch_size=args.batch_size, shuffle=False)

    # 2. Model
    print("🧠 Initializing U-Bit-SegNet (v3)...")
    depths = [int(x) for x in args.depths.split(',')]
    model = BitSegNet(
        img_size=args.img_size,
        in_chans=3,
        base_dim=args.base_dim,
        depths=depths
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model params: {total_params / 1e6:.2f}M")

    # 3. Optimizer
    if args.use_galore:
        print("🔧 Using GaLoreAdamW8bit optimizer for memory-efficient training.")
        # Separate Galore params from regular params (norm layers, biases)
        galore_params = []
        regular_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() >= 2 and 'norm' not in n and 'pos_embed' not in n:
                galore_params.append(p)
            else:
                regular_params.append(p)
                
        param_groups = [
            {'params': regular_params},
            {'params': galore_params, 'rank': 128, 'update_proj_gap': 200, 'scale': 0.25, 'proj_type': 'std'}
        ]
        optimizer = GaLoreAdamW(param_groups, lr=args.lr, weight_decay=0.01)
    else:
        print("🔧 Using standard AdamW optimizer.")
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    scaler = GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 4. Training Loop
    print("🚂 Starting training...")
    best_val_loss = float('inf')
    start_epoch = 1

    if args.resume and os.path.isfile(args.resume):
        print(f"🔄 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint['val_loss']
        print(f"   -> Resumed from epoch {start_epoch - 1} with Val Loss {best_val_loss:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        # -- TRAIN --
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for step, (images, masks) in enumerate(pbar):
            if isinstance(images, torch.Tensor):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                logits = model(images)
                loss = segmentation_loss(logits, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.1e}"})

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        
        # -- VALIDATION --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]", leave=False):
                if isinstance(images, torch.Tensor):
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                with autocast():
                    logits = model(images)
                    v_loss = segmentation_loss(logits, masks)
                val_loss += v_loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"📊 Epoch {epoch} complete | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            
        if epoch % 10 == 0 or epoch == args.epochs or is_best:
            ckpt_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'args': vars(args)
            }
            
            if epoch % 10 == 0 or epoch == args.epochs:
                ckpt_path = os.path.join(args.output_dir, f"bit_segnet_ep{epoch:03d}.pt")
                torch.save(ckpt_data, ckpt_path)
                print(f"💾 Saved epoch checkpoint to {ckpt_path}")
                
            if is_best:
                best_path = os.path.join(args.output_dir, "bit_segnet_best.pt")
                torch.save(ckpt_data, best_path)
                print(f"🌟 New Best Val Loss! Saved to {best_path}")

if __name__ == '__main__':
    main()
