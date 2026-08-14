import os
import argparse
import time
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import torchvision.transforms as transforms
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForZeroShotImageClassification

from bit_clip import BitCLIP

class CrossLingualDistillDataset(Dataset):
    def __init__(self, hf_dataset_name, tokenizer, transform=None):
        print(f"🌍 Loading HuggingFace Teacher Dataset ({hf_dataset_name})...")
        self.hf_data = load_dataset(hf_dataset_name, split="train")
        
        self.transform = transform
        self.tokenizer = tokenizer
        
        print(f"🇯🇵 Loading HuggingFace STAIR Captions (shunk031/STAIR-Captions)...")
        stair_dataset = load_dataset("shunk031/STAIR-Captions", split="train")
            
        self.stair_dict = {}
        for item in stair_dataset:
            img_id = item.get("image_id", item.get("id"))
            if img_id not in self.stair_dict:
                self.stair_dict[img_id] = []
            self.stair_dict[img_id].append(item["caption"])
            
        print(f"✅ Loaded {len(self.hf_data)} pre-computed items and mapped {len(self.stair_dict)} Japanese images.")

    def __len__(self):
        return len(self.hf_data)

    def __getitem__(self, idx):
        item = self.hf_data[idx]
        img_id = item.get("image_id", item.get("id"))
        
        # Load Image Directly from HuggingFace Dataset! (No manual download needed)
        # The 'image_bytes' column is automatically cast to a PIL Image by datasets library
        image = item["image_bytes"].convert("RGB")
            
        if self.transform:
            image = self.transform(image)
            
        # Get Japanese Text (STAIR)
        if img_id in self.stair_dict:
            ja_text = random.choice(self.stair_dict[img_id])
        else:
            ja_text = "画像"
            
        # Get English Text (Original COCO)
        en_text = item["caption"]
            
        # Tokenize Japanese (For Student)
        inputs_ja = self.tokenizer(
            ja_text, 
            padding="max_length", 
            max_length=64, 
            truncation=True, 
            return_tensors="pt"
        )
        input_ids_ja = inputs_ja["input_ids"].squeeze(0)
        
        # Tokenize English (For Teacher on-the-fly)
        inputs_en = self.tokenizer(
            en_text, 
            padding="max_length", 
            max_length=64, 
            truncation=True, 
            return_tensors="pt"
        )
        input_ids_en = inputs_en["input_ids"].squeeze(0)
        
        # Teacher Image Vector (SigLIP2 1152d)
        teacher_image_embed = torch.tensor(item["vector"])
            
        return {
            "image": image,
            "input_ids_ja": input_ids_ja,
            "input_ids_en": input_ids_en,
            "teacher_image_embed": teacher_image_embed
        }

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_dataset", type=str, default="jrmiller/coco-2017-siglip2-embeddings")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()

def distill_loss(student_embed, teacher_embed):
    mse = F.mse_loss(student_embed, teacher_embed)
    cos = 1.0 - F.cosine_similarity(student_embed, teacher_embed, dim=-1).mean()
    return mse + cos * 0.1

def main():
    args = get_args()
    
    is_distributed = False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        is_distributed = True
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        global_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    is_master = (global_rank == 0)

    if is_master:
        print(f"🚀 Starting Cross-Lingual Distillation on {device}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-256-multilingual")

    dataset = CrossLingualDistillDataset(
        args.hf_dataset, 
        tokenizer, 
        transform
    )
    
    if is_distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=(sampler is None), 
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Load Teacher Text Encoder for on-the-fly text embeddings
    # The dataset uses a 1152d SigLIP2 model (likely siglip2-large)
    if is_master:
        print("Loading Teacher Text Encoder (google/siglip2-large-patch16-256)...")
    teacher_model = AutoModelForZeroShotImageClassification.from_pretrained("google/siglip-large-patch16-256")
    teacher_text_model = teacher_model.text_model.to(device)
    teacher_text_model.eval()

    # Initialize BitCLIP with embed_dim=1152 to match Teacher
    model = BitCLIP(
        embed_dim=1152,       
        vocab_size=250000,   
        img_size=256,        
        patch_size=16,
        d_model=256,
        n_layers=6,
        d_state=32,
    ).to(device)

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            gradient_as_bucket_view=True
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    for epoch in range(args.epochs):
        if is_distributed:
            sampler.set_epoch(epoch)
            
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        for step, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            input_ids_ja = batch["input_ids_ja"].to(device)
            input_ids_en = batch["input_ids_en"].to(device)
            teacher_img = batch["teacher_image_embed"].to(device)
            
            optimizer.zero_grad()
            
            # Generate Teacher Text Vector on-the-fly (No gradients needed)
            with torch.no_grad():
                teacher_txt_features = teacher_text_model(input_ids_en).pooler_output
                teacher_txt = F.normalize(teacher_txt_features, dim=-1)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                base_model = model.module if is_distributed else model
                
                # Student processing
                student_img_features = base_model.image_encoder(images)
                student_txt_features = base_model.text_encoder(input_ids_ja)
                
                student_img_embeds = base_model.image_proj(student_img_features)
                student_txt_embeds = base_model.text_proj(student_txt_features)
                
                student_img_embeds = F.normalize(student_img_embeds, dim=-1)
                student_txt_embeds = F.normalize(student_txt_embeds, dim=-1)
                
                # MSE Distillation
                loss_img = distill_loss(student_img_embeds, teacher_img)
                loss_txt = distill_loss(student_txt_embeds, teacher_txt)
                loss = loss_img + loss_txt
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()

            if is_master and (step + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{step+1}/{len(dataloader)}] Loss: {loss.item():.4f} (Img: {loss_img.item():.4f}, Txt: {loss_txt.item():.4f})")

        if is_master:
            avg_loss = total_loss / len(dataloader)
            print(f"🎯 Epoch {epoch+1} Completed | Avg Distill Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.1f}s")
            
    if is_distributed:
        torch.distributed.destroy_process_group()
        
    if is_master:
        out_file = "bit_clip_student.pt"
        base_model = model.module if is_distributed else model
        torch.save(base_model.state_dict(), out_file)
        print(f"💾 Student model successfully saved to {out_file}!")

if __name__ == "__main__":
    main()
