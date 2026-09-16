import os
import argparse
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def save_split(split_name, out_dir, max_samples=None):
    img_dir = os.path.join(out_dir, split_name, 'images')
    mask_dir = os.path.join(out_dir, split_name, 'masks')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    print(f"Downloading {split_name} split to {out_dir}/{split_name}...")
    
    # We use streaming=True just in case, or we can load it normally if memory allows.
    # The dataset provides PIL Images directly if loaded normally.
    try:
        ds = load_dataset('nobg/DIS5K', split=split_name, trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load split {split_name}. Error: {e}")
        return

    count = 0
    for item in tqdm(ds, desc=f"Saving {split_name}"):
        if max_samples and count >= max_samples:
            break
            
        # Extract names without extension to avoid .jpg.png double extensions
        img_name = os.path.splitext(item['image_name'])[0]
        
        # Save image (convert to RGB)
        image = item['image'].convert('RGB')
        image.save(os.path.join(img_dir, f"{img_name}.jpg"))
        
        # Save mask (convert to L - grayscale)
        label = item['label'].convert('L')
        label.save(os.path.join(mask_dir, f"{img_name}.png"))
        
        count += 1
        
    print(f"Saved {count} samples for {split_name}.")

def main():
    parser = argparse.ArgumentParser(description="Download and prepare DIS5K dataset")
    parser.add_argument("--out_dir", type=str, default="./data/dis5k", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to download (for testing)")
    args = parser.parse_args()

    # DIS5K splits: DIS_TR (Train: ~3000), DIS_VD (Val: ~470)
    save_split('DIS_TR', args.out_dir, max_samples=args.max_samples)
    
    # Rename folder to 'train' to match typical dataloader expectations
    if os.path.exists(os.path.join(args.out_dir, 'DIS_TR')):
        os.rename(os.path.join(args.out_dir, 'DIS_TR'), os.path.join(args.out_dir, 'train'))

    save_split('DIS_VD', args.out_dir, max_samples=args.max_samples)
    
    # Rename folder to 'val' to match typical dataloader expectations
    if os.path.exists(os.path.join(args.out_dir, 'DIS_VD')):
        os.rename(os.path.join(args.out_dir, 'DIS_VD'), os.path.join(args.out_dir, 'val'))

    print(f"\n✅ DIS5K Preparation Complete! Data saved to {args.out_dir}")
    print(f"Directory structure:")
    print(f"  {args.out_dir}/train/images/")
    print(f"  {args.out_dir}/train/masks/")
    print(f"  {args.out_dir}/val/images/")
    print(f"  {args.out_dir}/val/masks/")

if __name__ == "__main__":
    main()
