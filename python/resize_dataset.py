import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Allow PIL to load huge images
Image.MAX_IMAGE_PIXELS = None

def resize_images_in_dir(img_dir, mask_dir, out_img_dir, out_mask_dir, size):
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)
    out_img_dir = Path(out_img_dir)
    out_mask_dir = Path(out_mask_dir)
    
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    
    images = list(img_dir.glob("*.*"))
    for img_path in tqdm(images, desc=f"Resizing {img_dir.parent.name}"):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
            
        mask_path = mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists():
            continue
            
        # Resize image
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize((size, size), Image.Resampling.BILINEAR)
        img_resized.save(out_img_dir / f"{img_path.stem}.jpg")
        
        # Resize mask
        mask = Image.open(mask_path).convert("L")
        mask_resized = mask.resize((size, size), Image.Resampling.NEAREST)
        mask_resized.save(out_mask_dir / f"{img_path.stem}.png")

def main():
    parser = argparse.ArgumentParser(description="Pre-resize datasets to speed up dataloading")
    parser.add_argument("--data_dir", type=str, required=True, help="Input data dir (e.g. data/dis5k)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output data dir (e.g. data/dis5k_256)")
    parser.add_argument("--size", type=int, default=256, help="Target size")
    args = parser.parse_args()
    
    for split in ['train', 'val']:
        in_split_dir = os.path.join(args.data_dir, split)
        if os.path.exists(in_split_dir):
            resize_images_in_dir(
                os.path.join(in_split_dir, 'images'),
                os.path.join(in_split_dir, 'masks'),
                os.path.join(args.out_dir, split, 'images'),
                os.path.join(args.out_dir, split, 'masks'),
                args.size
            )
            
    print(f"✅ Resize complete! Saved to {args.out_dir}")

if __name__ == "__main__":
    main()
