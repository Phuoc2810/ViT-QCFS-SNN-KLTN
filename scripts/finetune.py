import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import os
import sys

# Import config và hàm load data xịn từ dataset.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model_ann import VisionTransformer 
from src.dataset import get_dataloader # <--- Dùng cái này thay vì tự viết transform
from src.config import Config

def get_args():
    parser = argparse.ArgumentParser(description='Fine-tune Pruned ANN')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to best ANN checkpoint')
    parser.add_argument('--save_path', type=str, default='./checkpoints/ann_pruned_finetuned.pth')
    
    # --- THÊM DÒNG NÀY ĐỂ CHỌN DATASET ---
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Choose dataset')
    
    parser.add_argument('--pruning_ratio', type=float, default=0.2, help='Amount of sparsity (e.g. 0.2 = 20%)')
    parser.add_argument('--epochs', type=int, default=15, help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Low learning rate for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=128)
    
    # Kiến trúc Model
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

def main():
    torch.autograd.set_detect_anomaly(True)
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Dataset: {args.dataset.upper()}")

    # 1. Load Data (Dùng hàm chung từ dataset.py)
    # Lưu ý: Hàm get_dataloader phải return train_loader, test_loader, num_classes
    # (Nếu dataset.py chưa return num_classes thì bạn tự gán thủ công bên dưới)
    try:
        train_loader, test_loader, num_classes = get_dataloader(batch_size=args.batch_size, dataset_name=args.dataset)
    except ValueError: 
        # Fallback nếu hàm get_dataloader của bạn chưa sửa return num_classes
        train_loader, test_loader = get_dataloader(batch_size=args.batch_size, dataset_name=args.dataset)
        num_classes = 100 if args.dataset == 'cifar100' else 10

    print(f"Data loaded. Num classes: {num_classes}")

    # 2. Khởi tạo & Load Model Gốc
    print(f"Loading checkpoint from {args.checkpoint_path}")
    
    # QUAN TRỌNG: Truyền đúng num_classes vào model
    model = VisionTransformer(
        dim=args.embed_dim, depth=args.depth, heads=args.heads, 
        num_classes=num_classes, # <--- Không hardcode số 10 nữa
        T=0, L=args.L
    ).to(device)
    
    # Load Checkpoint an toàn (tránh lỗi lệch key)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Lọc bỏ các key không khớp (ví dụ nếu checkpoint cũ là 10 class mà giờ load vào model 100 class)
    # Tuy nhiên, Pruning thường dùng chính checkpoint của dataset đó nên không lo.
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"⚠️ Cảnh báo: Lỗi key khi load model (có thể do sai số class): {e}")
        # Nếu cần thiết thì bỏ qua strict loading (chỉ dùng khi transfer learning)
        # model.load_state_dict(state_dict, strict=False) 

    # Reset về chế độ ANN
    print("🔧 Đang cưỡng chế toàn bộ mạng về T=0 (Chế độ ANN)...")
    for module in model.modules():
        if hasattr(module, 'T'): module.T = 0
        if hasattr(module, 'mem'): module.mem = None
            
    # 3. Áp dụng Pruning
    with torch.no_grad():
        pruning_params = apply_pruning(model, args.pruning_ratio)
    
    # 4. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Biến theo dõi
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    # 5. Training Loop
    model.train()
    print(f"🚀 Start Fine-tuning on {args.dataset.upper()}...")
    
    for epoch in range(args.epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            reset_model(model)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"   ⭐ New Best Accuracy!")
        else:
            patience_counter += 1
            print(f"   ⚠️ No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print("🛑 Early Stopping!")
                break

    # 6. Lưu kết quả
    print("\n" + "="*50)
    if best_model_state is not None:
        print("🔄 Loading best model...")
        model.load_state_dict(best_model_state)
    
    make_pruning_permanent(pruning_params)
    
    # Kiểm tra độ thưa
    total_zeros = sum(torch.sum(m.weight == 0).item() for m in model.modules() if isinstance(m, nn.Linear))
    total_params = sum(m.weight.nelement() for m in model.modules() if isinstance(m, nn.Linear))
    print(f"✅ Final Sparsity: {100. * total_zeros / total_params:.2f}%")

    # Tạo thư mục và lưu
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, args.save_path)
    print(f"💾 Saved to {args.save_path}")

if __name__ == '__main__':
    main()