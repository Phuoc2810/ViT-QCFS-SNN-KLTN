import torch
import argparse
from tqdm import tqdm
import time

# Import classes
from src.config import Config
from src.model_ann import VisionTransformer
from src.model_snn import SpikeVisionTransformer
from src.dataset import get_dataloader

def get_args():
    parser = argparse.ArgumentParser(description='Convert ANN to SNN and Evaluate')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to trained ANN checkpoint (.pth)')
    parser.add_argument('--timesteps', type=str, default="4,8,16", help='Comma separated timesteps list (e.g. 4,8,16)')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable tqdm')
    parser.add_argument('--embed_dim', type=int, default=192)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--heads', type=int, default=3)
    return parser.parse_args()

def transfer_weights(ann_model, snn_model):
    """
    Chuyển trọng số từ ANN sang SNN, xử lý việc đổi tên các layer MLP
    """
    ann_state = ann_model.state_dict()
    snn_state = snn_model.state_dict()
    new_snn_state = {}

    print("Transferring weights...")
    for name, param in ann_state.items():
        if name in snn_state:
            new_snn_state[name] = param
        else:
            # Xử lý mapping tên layer khác nhau giữa ANN và SNN
            snn_name = None
            if ".mlp.0." in name: 
                snn_name = name.replace(".mlp.0.", ".mlp_fc1.")
            elif ".mlp.3." in name:
                snn_name = name.replace(".mlp.3.", ".mlp_fc2.")
            
            if snn_name and snn_name in snn_state:
                new_snn_state[snn_name] = param
                
    # Load dictionary mới vào SNN model
    snn_model.load_state_dict(new_snn_state, strict=True)
    print("Weight transfer complete!")

def evaluate_snn(model, loader, device, T, disable_tqdm=False):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    
    # Cập nhật Timestep cho model
    model.T = T
    # Cập nhật T cho các lớp IF bên trong (nếu cần thiết kế lại hàm set_T, hoặc khởi tạo lại model)
    # Ở đây giả định khởi tạo lại model cho chắc chắn trong vòng lặp main
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Evaluating T={T}", disable=disable_tqdm):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    acc = 100. * correct / total
    duration = time.time() - start_time
    return acc, duration

def main():
    args = get_args()
    device = Config.DEVICE
    timesteps_list = [int(t) for t in args.timesteps.split(',')]
    
    # 1. Load ANN
    print(f"Loading ANN from {args.checkpoint_path}")
    ann_model = VisionTransformer(
        dim=args.embed_dim, 
        depth=args.depth, 
        heads=args.heads, 
        num_classes=10
    )
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    ann_model.load_state_dict(checkpoint['model_state_dict'])
    ann_model.to(device)
    ann_model.eval()
    
    # 2. Prepare Data
    _, test_loader = get_dataloader(batch_size=64) # Batch nhỏ hơn chút cho SNN đỡ tốn VRAM
    
    print("\n--- Starting SNN Conversion & Evaluation ---")
    print(f"{'Timesteps':<10} | {'Accuracy (%)':<15} | {'Time (s)':<10}")
    print("-" * 40)
    
    for T in timesteps_list:
        # Khởi tạo SNN mới với T tương ứng
        snn_model = SpikeVisionTransformer(
            dim=args.embed_dim, # Dùng args
            depth=args.depth, 
            heads=args.heads, 
            T=T, L=8
        ).to(device)
        # Copy weights
        transfer_weights(ann_model, snn_model)
        
        # Đánh giá
        acc, duration = evaluate_snn(snn_model, test_loader, device, T, disable_tqdm=args.no_progress_bar)
        print(f"{T:<10} | {acc:<15.2f} | {duration:<10.2f}")

if __name__ == "__main__":
    main()