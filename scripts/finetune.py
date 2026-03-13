import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import os
import sys

# Import config và hàm load data
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model_ann import VisionTransformer 
from src.dataset import get_dataloader 
from src.config import Config

def get_args():
    parser = argparse.ArgumentParser(description='Fine-tune Pruned ANN')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to best ANN checkpoint')
    parser.add_argument('--save_path', type=str, default='./checkpoints/ann_pruned_finetuned.pth')
    
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Choose dataset')
    
    parser.add_argument('--pruning_ratio', type=float, default=0.2, help='Amount of sparsity (e.g. 0.2 = 20%)')
    parser.add_argument('--epochs', type=int, default=15, help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Low learning rate for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=128)
    
    parser.add_argument('--embed_dim', type=int, default=192)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--heads', type=int, default=3)
    parser.add_argument('--L', type=int, default=4, help='Quantization levels')
    parser.add_argument('--patience', type=int, default=10)
    
    return parser.parse_args()

def apply_pruning(model, amount):
    print(f"✂️ Applying pruning mask with ratio: {amount}...")
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Không prune layer cuối cùng (head) để giữ accuracy ổn định
            if "head" in name: 
                continue
            parameters_to_prune.append((module, 'weight'))
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return parameters_to_prune

def make_pruning_permanent(parameters_to_prune):
    print("🔒 Making pruning permanent...")
    for module, name in parameters_to_prune:
        prune.remove(module, name)

def reset_model(model):
    for name, module in model.named_modules():
        if hasattr(module, 'mem'):
            module.mem = None

# --- Thêm hàm Validation riêng biệt ---
def validate(model, test_loader, criterion, device):
    model.eval() # Quan trọng: Chuyển sang chế độ đánh giá
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            reset_model(model)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    return avg_loss, acc

def main():
    torch.autograd.set_detect_anomaly(True)
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Dataset: {args.dataset.upper()}")

    # 1. Load Data
    try:
        train_loader, test_loader, num_classes = get_dataloader(batch_size=args.batch_size, dataset_name=args.dataset)
    except ValueError: 
        train_loader, test_loader = get_dataloader(batch_size=args.batch_size, dataset_name=args.dataset)
        num_classes = 100 if args.dataset == 'cifar100' else 10

    print(f"Data loaded. Num classes: {num_classes}")

    # 2. Khởi tạo & Load Model
    print(f"Loading checkpoint from {args.checkpoint_path}")
    model = VisionTransformer(
        dim=args.embed_dim, depth=args.depth, heads=args.heads, 
        num_classes=num_classes, 
        T=0, L=args.L
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"⚠️ Cảnh báo key: {e}")

    # Reset về ANN
    for module in model.modules():
        if hasattr(module, 'T'): module.T = 0
        if hasattr(module, 'mem'): module.mem = None
            
    # 3. Apply Pruning
    with torch.no_grad():
        pruning_params = apply_pruning(model, args.pruning_ratio)
    
    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    # 5. Training Loop
    print(f"🚀 Start Fine-tuning on {args.dataset.upper()}...")
    
    for epoch in range(args.epochs):
        model.train() # Chuyển về chế độ train
        train_loss = 0
        correct = 0
        total = 0
        
        # --- TRAIN ---
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            reset_model(model)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATE (TEST) ---
        # Đây là bước quan trọng nhất để biết acc thực tế
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.2f}%") # <--- In ra Val Acc

        # Save Best dựa trên VAL ACC (không phải Train Acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  ⭐ New Best Accuracy! (Saved)")
        else:
            patience_counter += 1
            print(f"  ⚠️ No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print("🛑 Early Stopping!")
                break

    # 6. Lưu kết quả
    print("\n" + "="*50)
    print(f"Best Val Accuracy: {best_acc:.2f}% at Epoch {best_epoch}")
    
    if best_model_state is not None:
        print("🔄 Loading best model...")
        model.load_state_dict(best_model_state)
    
    make_pruning_permanent(pruning_params)
    
    total_zeros = sum(torch.sum(m.weight == 0).item() for m in model.modules() if isinstance(m, nn.Linear))
    total_params = sum(m.weight.nelement() for m in model.modules() if isinstance(m, nn.Linear))
    print(f"✅ Final Sparsity: {100. * total_zeros / total_params:.2f}%")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'args': args}, args.save_path)
    print(f"💾 Saved to {args.save_path}")

if __name__ == '__main__':
    main()