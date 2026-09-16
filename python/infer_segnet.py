import argparse
import os
import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path

from bit_segnet import BitSegNet

def get_args_parser():
    parser = argparse.ArgumentParser('Bit-SegNet Inference', add_help=False)
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained checkpoint (.pt)')
    parser.add_argument('--output', type=str, default='output.png', help='Path to save the output RGBA image')
    parser.add_argument('--alpha_thresh', type=float, default=0.5, help='Threshold for binarizing the mask (0-1)')
    parser.add_argument('--soft_mask', action='store_true', help='Use soft alpha matting instead of binary cut')
    return parser

def main():
    parser = argparse.ArgumentParser('Bit-SegNet Inference', parents=[get_args_parser()])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    # 1. Load Checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    print(f"📦 Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    ckpt_args = checkpoint.get('args', {})
    
    # 2. Initialize Model
    # Fallback to default values if not in checkpoint (for older saves)
    img_size = ckpt_args.get('img_size', 256)
    base_dim = ckpt_args.get('base_dim', 64)
    depths_str = ckpt_args.get('depths', '1,1,2,1')
    depths = [int(x) for x in depths_str.split(',')]

    print("🧠 Initializing U-Bit-SegNet (v3)...")
    model = BitSegNet(
        img_size=img_size,
        in_chans=3,
        base_dim=base_dim,
        depths=depths
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded successfully.")

    # 3. Load and Preprocess Image
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Input image not found: {args.image}")
    
    orig_img = Image.open(args.image).convert("RGB")
    orig_w, orig_h = orig_img.size
    print(f"🖼️ Loaded image: {args.image} ({orig_w}x{orig_h})")

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = transform(orig_img).unsqueeze(0).to(device)

    # 4. Run Inference
    print("🔮 Running inference...")
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            logits = model(img_tensor)
            probs = torch.sigmoid(logits) # (1, 1, img_size, img_size)

    # 5. Postprocess Mask
    mask = probs[0, 0].cpu()
    if not args.soft_mask:
        mask = (mask > args.alpha_thresh).float()
    
    # Convert mask to PIL Image and resize back to original dimensions
    mask_img = T.ToPILImage()(mask)
    mask_img = mask_img.resize((orig_w, orig_h), resample=Image.Resampling.BILINEAR)

    # 6. Apply Mask to Original Image (Create RGBA)
    orig_img.putalpha(mask_img)

    # 7. Save Output
    orig_img.save(args.output)
    print(f"🎉 Background removed! Saved output to {args.output}")

if __name__ == '__main__':
    main()
