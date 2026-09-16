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
        
        # Basic transforms (RGB images)
        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Mask transforms (Grayscale)
        self.mask_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

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

        # Data augmentation could be added here (random flip, color jitter, etc.)
        # For simplicity, we just apply standard resize and normalize
        
        image_t = self.img_transform(image)
        mask_t = self.mask_transform(mask)
        
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
    parser.add_argument('--patch_size', default=16, type=int, help='patch size for spatial SSM')
    parser.add_argument('--d_model', default=192, type=int, help='transformer channel dimension')
    parser.add_argument('--n_layers', default=6, type=int, help='number of SSM layers')
    
    # Training parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--use_galore', action='store_true', help='Use GaLore optimizer for low memory footprint')
    parser.add_argument('--num_workers', default=4, type=int)
    
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
        dataset = ImageMaskDataset(args.data_dir, img_size=args.img_size, is_train=True)
        if len(dataset) == 0:
            raise FileNotFoundError("Dataset is empty.")
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                                num_workers=args.num_workers, pin_memory=True, drop_last=True)
        print(f"✅ Found {len(dataset)} image-mask pairs.")
    except FileNotFoundError:
        print(f"⚠️ Warning: Dataset not found at {args.data_dir}. Creating a DUMMY dataset for testing.")
        # Create a dummy dataloader for dry-run testing
        dataset = [(torch.randn(3, args.img_size, args.img_size), 
                    (torch.rand(1, args.img_size, args.img_size) > 0.5).float()) for _ in range(64)]
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 2. Model
    print("🧠 Initializing Bit-SegNet...")
    model = BitSegNet(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        d_model=args.d_model,
        n_layers=args.n_layers
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
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
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

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.1e}"})

        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"📊 Epoch {epoch} complete | Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.output_dir, f"bit_segnet_ep{epoch:03d}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': vars(args)
            }, ckpt_path)
            print(f"💾 Saved checkpoint to {ckpt_path}")

if __name__ == '__main__':
    main()
