import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Import từ thư mục src
from src.config import Config
from src.dataset import get_dataloader
from src.model_ann import VisionTransformer
from src.utils import set_seed

def get_args():
    parser = argparse.ArgumentParser(description='Train ANN Vision Transformer')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable tqdm progress bar')
    parser.add_argument('--save_path', type=str, default='./checkpoints/best_ann.pth', help='Path to save best model')
    parser.add_argument('--L', type=int, default=8, help='Quantization Levels (QCFS)')
    parser.add_argument('--T', type=int, default=0, help='Timesteps (0 for ANN, >0 for SNN training)')
    parser.add_argument('--depth', type=int, default=12, help='Model Depth')
    parser.add_argument('--embed_dim', type=int, default=192, help='Embedding Dimension')
    parser.add_argument('--heads', type=int, default=3, help='Number of Heads')
    
    return parser.parse_args()

def train(model, loader, criterion, optimizer, device, disable_tqdm=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Dùng tqdm để hiện thanh tiến trình
    pbar = tqdm(loader, desc="Training", leave=False, disable=disable_tqdm)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if not disable_tqdm:
            pbar.set_postfix({'Loss': running_loss/total, 'Acc': 100.*correct/total})
        
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device, disable_tqdm=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False, disable=disable_tqdm):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(loader), 100. * correct / total

def main():
    args = get_args()
    set_seed(Config.SEED)
    device = Config.DEVICE
    
    # Tạo thư mục chứa checkpoint nếu chưa có
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Load Data
    train_loader, test_loader = get_dataloader(batch_size=args.batch_size)
    
    # Khởi tạo Model
    print("Creating ANN Model: Depth={args.depth}, Heads={args.heads}, L={args.L}")
    model = model = VisionTransformer(
        dim=args.embed_dim, 
        depth=args.depth, 
        heads=args.heads, 
        num_classes=10,
        T=args.T,  # Thường ANN thì T=0
        L=args.L   # Quan trọng cho QCFS
    ).to(device)
     
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_acc = 0.0
    
    print(f"Start training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, disable_tqdm=args.no_progress_bar)
        val_loss, val_acc = validate(model, test_loader, criterion, device, disable_tqdm=args.no_progress_bar)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"--> New Best Accuracy! Saving model to {args.save_path}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, args.save_path)

if __name__ == "__main__":
    main()