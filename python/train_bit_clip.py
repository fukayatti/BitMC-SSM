import os
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms
from transformers import GPT2TokenizerFast

from bit_clip import BitCLIP, bit_clip_loss

def get_args():
    parser = argparse.ArgumentParser(description="Train Bit-CLIP with DDP")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")
    return parser.parse_args()

def main():
    args = get_args()
    
    # DDP Initialization
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
        print(f"🚀 Starting Bit-CLIP DDP Training on {device}")
        if is_distributed:
            print(f"🌍 Distributed Mode: Enabled (World Size: {os.environ['WORLD_SIZE']})")

    # Data Preparation
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if is_master:
        # Download datasets only on master to prevent conflicts
        torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    if is_distributed:
        torch.distributed.barrier()

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

    if is_distributed:
        train_sampler = DistributedSampler(trainset)
        test_sampler = DistributedSampler(testset, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    trainloader = DataLoader(
        trainset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler, 
        drop_last=True,
        num_workers=2,
        pin_memory=True
    )
    
    testloader = DataLoader(
        testset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        sampler=test_sampler,
        num_workers=2,
        pin_memory=True
    )

    # Prompts for CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    class_prompts = [f"a photo of a {c}." for c in classes]
    prompt_tokens = tokenizer(class_prompts, padding=True, return_tensors='pt').input_ids.to(device)

    # Initialize Model
    model = BitCLIP(
        embed_dim=256,
        vocab_size=50257,
        img_size=32,
        patch_size=4,  # 32/4 = 8x8 (64 patches)
        d_model=192,
        n_layers=4,
        d_state=32,
    ).to(device)

    if is_master:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"🧠 Bit-CLIP Parameters: {total_params / 1e6:.2f}M")

    # DDP Wrapper
    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            gradient_as_bucket_view=True
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # Training Loop
    for epoch in range(args.epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)
            
        model.train()
        total_loss = 0.0
        start_time = time.time()
        
        for step, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            texts = prompt_tokens[labels]
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                if is_distributed:
                    logits_per_image, logits_per_text = model.module(images, texts)
                else:
                    logits_per_image, logits_per_text = model(images, texts)
                loss = bit_clip_loss(logits_per_image, logits_per_text)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()

            if is_master and (step + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{step+1}/{len(trainloader)}] Loss: {loss.item():.4f}")

        if is_distributed:
            loss_tensor = torch.tensor(total_loss, device=device)
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / (len(trainloader) * int(os.environ["WORLD_SIZE"]))
        else:
            avg_loss = total_loss / len(trainloader)

        if is_master:
            print(f"🎯 Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f} | Time: {time.time() - start_time:.1f}s")
            
    # Evaluation (Zero-Shot Classification)
    if is_master:
        print("🔍 Running Zero-Shot Classification on Test Set...")
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        base_model = model.module if is_distributed else model
        text_features = base_model.text_encoder(prompt_tokens)
        text_embeds = base_model.text_proj(text_features)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            image_features = base_model.image_encoder(images)
            image_embeds = base_model.image_proj(image_features)
            image_embeds = F.normalize(image_embeds, dim=-1)
            
            logits = base_model.logit_scale.exp() * image_embeds @ text_embeds.t()
            predicted = logits.argmax(dim=-1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if is_distributed:
        correct_tensor = torch.tensor(correct, device=device)
        total_tensor = torch.tensor(total, device=device)
        torch.distributed.all_reduce(correct_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)
        correct = correct_tensor.item()
        total = total_tensor.item()

    if is_master:
        accuracy = 100 * correct / total
        print(f"\\n🎉 Zero-Shot Accuracy on CIFAR-10: {accuracy:.2f}%")
        
    if is_distributed:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
