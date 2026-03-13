import torch
import argparse
from tqdm import tqdm
import time
import sys
import os

# Import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import Config
from src.dataset import get_dataloader 

# Import cả 2 model
from src.model_ann import VisionTransformer
from src.model_snn import SpikeVisionTransformer  # <--- Import model SNN riêng

def get_args():
    parser = argparse.ArgumentParser(description='Convert ANN to SNN and Evaluate')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to trained ANN checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--timesteps', type=str, default="4,8,16", help='Comma separated T (e.g. 4,8,16)')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable tqdm')
    
    # Optional args (Auto-detect từ checkpoint sẽ ưu tiên hơn)
    parser.add_argument('--embed_dim', type=int, default=None)
    parser.add_argument('--depth', type=int, default=None)
    parser.add_argument('--heads', type=int, default=None)
    parser.add_argument('--L', type=int, default=None)
    
    return parser.parse_args()

def evaluate_snn(model, loader, device, T, disable_tqdm=False):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    
    # Reset spike count buffer (nếu có)
    for m in model.modules():
        if hasattr(m, 'spike_count'):
            m.spike_count.zero_()

    with torch.no_grad():
        iterator = tqdm(loader, desc=f"Evaluating T={T}", disable=disable_tqdm)
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device)
            
            # Lưu ý: Model SNN của bạn đã tự động handle việc repeat ảnh 
            # và merge dimension trong hàm forward() -> x.unsqueeze(0).repeat(...)
            # Nên ta cứ đưa ảnh tĩnh [B, C, H, W] vào là được.
            
            outputs = model(images) 
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    acc = 100. * correct / total
    duration = time.time() - start_time
    
    return acc, duration

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timesteps_list = [int(t) for t in args.timesteps.split(',')]
    
    print(f"📂 Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    
    # --- 1. AUTO DETECT CONFIG ---
    saved_args = checkpoint.get('args', None)
    if saved_args:
        print("💡 Detected configuration from checkpoint!")
        embed_dim = saved_args.embed_dim
        depth = saved_args.depth
        heads = saved_args.heads
        L = saved_args.L
        num_classes = checkpoint.get('num_classes', 100 if args.dataset == 'cifar100' else 10)
    else:
        print("⚠️ Using command line arguments.")
        embed_dim = args.embed_dim if args.embed_dim else 192
        depth = args.depth if args.depth else 12
        heads = args.heads if args.heads else 3
        L = args.L if args.L else 8
        num_classes = 100 if args.dataset == 'cifar100' else 10

    print(f"   Model Config: Dim={embed_dim}, Depth={depth}, Heads={heads}, L={L}, Classes={num_classes}")

    # --- 2. PREPARE DATA ---
    try:
        _, test_loader, _ = get_dataloader(batch_size=64, dataset_name=args.dataset)
    except:
        _, test_loader = get_dataloader(batch_size=64, dataset_name=args.dataset)

    # --- 3. CONVERSION LOOP ---
    print("\n--- SNN Evaluation Report ---")
    print(f"{'T':<5} | {'Acc (%)':<10} | {'Time (s)':<10}")
    print("-" * 30)
    
    for T in timesteps_list:
        # Khởi tạo SNN Model (Dùng class SpikeVisionTransformer)
        snn_model = SpikeVisionTransformer(
            img_size=32, 
            patch_size=4, 
            num_classes=num_classes,
            dim=embed_dim, 
            depth=depth, 
            heads=heads, 
            mlp_ratio=4., # Mặc định theo config của bạn
            T=T, 
            L=L
        ).to(device)
        
        # --- LOAD WEIGHTS ---
        # Logic: Dù tên class Block khác nhau (Block vs SpikeBlock)
        # Nhưng cấu trúc bên trong (tên biến self.attn, self.mlp...) giống hệt nhau
        # Nên load_state_dict vẫn hoạt động hoàn hảo!
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Load weights (Bỏ qua strict để tránh lỗi nhỏ nếu có biến thừa)
        msg = snn_model.load_state_dict(state_dict, strict=False)
        
        # --- EVALUATE ---
        acc, duration = evaluate_snn(snn_model, test_loader, device, T, disable_tqdm=args.no_progress_bar)
        print(f"{T:<5} | {acc:<10.2f} | {duration:<10.2f}")

if __name__ == "__main__":
    main()