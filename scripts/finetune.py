import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Import model của bạn (Đảm bảo đường dẫn đúng)
# Nếu bạn để model trong src/model.py thì sửa lại import cho phù hợp
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model_ann import VisionTransformer 

def get_args():
    parser = argparse.ArgumentParser(description='Fine-tune Pruned ANN')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to best ANN checkpoint')
    parser.add_argument('--save_path', type=str, default='./checkpoints/ann_pruned_finetuned.pth')
    parser.add_argument('--pruning_ratio', type=float, default=0.2, help='Amount of sparsity (e.g. 0.2 = 20%)')
    parser.add_argument('--epochs', type=int, default=15, help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Low learning rate for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=128)
    
    # Kiến trúc Model (Phải khớp với model cũ)
    parser.add_argument('--embed_dim', type=int, default=192)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--heads', type=int, default=3)
    
    return parser.parse_args()

def apply_pruning(model, amount):
    """
    Áp dụng Pruning nhưng KHÔNG remove mask ngay.
    Việc giữ mask giúp PyTorch biết những trọng số nào cần giữ là 0 trong lúc train.
    """
    print(f"✂️ Applying pruning mask with ratio: {amount}...")
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Bỏ qua lớp classifier head để giữ độ chính xác cao nhất
            if "head" in name: 
                continue
            parameters_to_prune.append((module, 'weight'))
    
    # Dùng Global Unstructured Pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return parameters_to_prune

def make_pruning_permanent(parameters_to_prune):
    """
    Sau khi train xong, ta 'ép' mask vào trọng số thật để lưu file cho gọn.
    """
    print("🔒 Making pruning permanent...")
    for module, name in parameters_to_prune:
        prune.remove(module, name)

def reset_model(model):
    # Duyet qua các module để reset bộ nhớ mem nếu có
    for name, module in model.named_modules():
        # Neu module co bien 'mem', reset no ve None
        if hasattr(module, 'mem'):
            module.mem = None

def main():
    torch.autograd.set_detect_anomaly(True)
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Data (CIFAR-10)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(), # Dùng Augmentation nhẹ
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # 2. Khởi tạo & Load Model Gốc
    print(f"Loading checkpoint from {args.checkpoint_path}")
    model = VisionTransformer(
        dim=args.embed_dim, depth=args.depth, heads=args.heads, num_classes=10, T=0
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print("🔧 Đang cưỡng chế toàn bộ mạng về T=0 (Chế độ ANN)...")
    count = 0
    for name, module in model.named_modules():
        # Kiểm tra xem module có thuộc tính T không
        if hasattr(module, 'T'):
            module.T = 0 # Ép về 0
            count += 1
        # Reset luôn bộ nhớ mem nếu có
        if hasattr(module, 'mem'):
            module.mem = None
            
    print(f"✅ Đã reset T=0 cho {count} module. Mạng đã sạch sẽ!")
    
    # 3. Áp dụng Pruning (Tạo Mask)
    # Bọc trong no_grad để không dính graph vào vòng lặp train
    with torch.no_grad():
        pruning_params = apply_pruning(model, args.pruning_ratio)
    
    # 4. Setup Optimizer cho Fine-tuning
    # Lưu ý: Learning Rate phải RẤT NHỎ (1e-5 hoặc 5e-5) để không phá hỏng weight đang tốt
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
     

    # 5. Training Loop (Fine-tuning)
    model.train()
    print(f"🚀 Start Fine-tuning for {args.epochs} epochs...")
    
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

    # 6. Chốt đơn (Lưu model)
    # Trước khi lưu phải remove mask để biến 0 ảo thành 0 thật
    make_pruning_permanent(pruning_params)
    
    # Kiểm tra độ thưa (Sparsity)
    total_zeros = 0
    total_params = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            total_zeros += torch.sum(m.weight == 0).item()
            total_params += m.weight.nelement()
    print(f"✅ Final Model Sparsity: {100. * total_zeros / total_params:.2f}%")

    # Lưu file
    torch.save({'model_state_dict': model.state_dict()}, args.save_path)
    print(f"💾 Model saved to {args.save_path}")

if __name__ == '__main__':
    main()