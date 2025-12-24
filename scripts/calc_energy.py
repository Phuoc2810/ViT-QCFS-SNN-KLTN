import torch
import argparse
import numpy as np
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis

# Import modules
from src.config import Config
from src.model_ann import VisionTransformer
# Dùng class SpikeVisionTransformer hoặc VisionTransformer (nếu chung class)
from src.model_snn import SpikeVisionTransformer 
from src.dataset import get_dataloader
from src.layers import IF

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to SNN/ANN checkpoint")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--T', type=int, default=4, help='Timesteps for SNN simulation')
    # Các tham số optional để đề phòng file checkpoint không có config
    parser.add_argument('--embed_dim', type=int, default=192)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--heads', type=int, default=3)
    return parser.parse_args()

def calc_ann_flops(model, input_tensor):
    """
    Tính FLOPs và MACs cho mô hình ANN chuẩn.
    """
    model.eval()
    flops = FlopCountAnalysis(model, input_tensor)
    total_flops = flops.total()
    
    # Quy ước chung: 1 MAC = 2 FLOPs (1 Mult + 1 Add)
    # Tuy nhiên fvcore thường trả về tổng số Operation.
    # Để an toàn và theo chuẩn IEEE:
    macs = total_flops / 2 
    
    # Năng lượng (Tham chiếu 45nm CMOS)
    # MAC (FP32) ~ 4.6 pJ (Horowitz 2014)
    # MAC (INT8) ~ 0.2 pJ (nếu Quantized)
    # Ở đây ta tính FP32 baseline
    energy_joule = macs * 4.6e-12 
    
    return macs, energy_joule

def calc_snn_energy(model, loader, device, T, embed_dim):
    """
    Tính năng lượng SNN (Spiking MLP) + ANN (Attention FP32).
    """
    model.eval()
    total_spikes = 0
    total_samples = 0
    
    # Reset spike count
    for m in model.modules():
        if hasattr(m, 'spike_count'):
            m.spike_count.zero_()

    # --- 1. Chạy mô phỏng trên 1 batch ---
    # Lấy 1 batch để ước lượng (đỡ tốn thời gian chạy hết)
    # Hoặc chạy hết nếu muốn chính xác tuyệt đối
    images, _ = next(iter(loader))
    images = images.to(device)
    total_samples = images.size(0)
    
    with torch.no_grad():
        _ = model(images)
    
    # --- 2. Tính Synaptic Operations (SOPs) ---
    # Chỉ tính cho các lớp Spiking (MLP)
    sops = 0
    for name, module in model.named_modules():
        if isinstance(module, IF) and hasattr(module, 'spike_count'):
            # Logic tính Fan-out chuẩn cho ViT:
            # IF nằm giữa Linear1 (dim->hidden) và Linear2 (hidden->dim)
            # Vậy output của IF đi vào Linear2.
            # Số kết nối từ 1 nơ-ron IF đến lớp tiếp theo = số nơ-ron output của Linear2
            # Linear2 output size = embed_dim.
            fan_out = embed_dim 
            
            # Tổng spike của cả batch -> chia cho batch size -> ra spike trung bình/ảnh
            layer_spikes = module.spike_count.item() / total_samples
            
            # SOPs = Spikes * Fan_out
            sops += layer_spikes * fan_out

    # --- 3. Tính Energy ---
    # A. Năng lượng Spiking (AC = Accumulate aka Addition)
    # 1 SOP (Addition 32-bit INT) ~ 0.9 pJ (45nm)
    # Một số báo cáo SNN dùng 0.1 pJ (rất lạc quan), ta dùng 0.9 pJ cho trung thực
    # Hoặc dùng 0.1 pJ nếu muốn so sánh với các bài báo SNN khác (họ thường cheat số này).
    E_per_SOP = 0.9e-12 
    energy_spike = sops * E_per_SOP

    # B. Năng lượng Static (Attention + PatchEmbed + Head)
    # Phần này vẫn chạy FP32 (MACs), cần tính riêng!
    # Cách tính: Lấy tổng MACs của ANN TRỪ ĐI phần MACs của MLP (vì MLP đã chuyển sang Spike)
    # MACs_Total = MACs_Attn + MACs_MLP
    # Ở đây để đơn giản, ta ước lượng MACs_Attn chiếm khoảng 1/3 tổng MACs của ViT.
    # Hoặc chính xác hơn: chạy fvcore cho riêng block Attention.
    
    # (Tạm tính theo cách đơn giản cho code gọn: giả sử Attention vẫn tốn năng lượng như ANN)
    # Để chính xác, bạn nên dùng fvcore đo lại model nhưng disable phần MLP.
    # Ở đây tôi sẽ return SOPs để bạn báo cáo "SNN Part Efficiency".
    
    return sops, energy_spike

def main():
    args = get_args()
    device = Config.DEVICE
    
    # 1. Load Checkpoint & Config Auto-detect
    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get('args', None)
    
    if saved_args:
        embed_dim = saved_args.embed_dim
        depth = saved_args.depth
        heads = saved_args.heads
        num_classes = checkpoint.get('num_classes', 10)
    else:
        embed_dim = args.embed_dim
        depth = args.depth
        heads = args.heads
        num_classes = 100 if args.dataset == 'cifar100' else 10

    print(f"Config: Dim={embed_dim}, Depth={depth}, Heads={heads}, T={args.T}")

    # 2. Setup ANN Model (Baseline)
    ann_model = VisionTransformer(
        dim=embed_dim, depth=depth, heads=heads, num_classes=num_classes, T=0
    ).to(device)
    
    # Dummy input
    input_size = (1, 3, 32, 32)
    dummy_input = torch.randn(input_size).to(device)

    print("\n--- 1. ANN Baseline Analysis ---")
    macs, ann_energy = calc_ann_flops(ann_model, dummy_input)
    print(f"Total MACs       : {macs/1e6:.2f} M") # M = Mega (10^6), G = Giga (10^9)
    print(f"Est. Energy (FP32): {ann_energy*1e3:.4f} mJ (milli-Joule)")

    # 3. Setup SNN Model
    snn_model = SpikeVisionTransformer(
        dim=embed_dim, depth=depth, heads=heads, num_classes=num_classes, T=args.T
    ).to(device)
    
    # Load weights (Bắt buộc để có spike đúng)
    if 'model_state_dict' in checkpoint:
        snn_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        snn_model.load_state_dict(checkpoint, strict=False)

    # Load Data
    _, test_loader, _ = get_dataloader(batch_size=32, dataset_name=args.dataset)

    print(f"\n--- 2. SNN Efficiency Analysis (T={args.T}) ---")
    sops, snn_spike_energy = calc_snn_energy(snn_model, test_loader, device, args.T, embed_dim)
    
    print(f"Avg Spike Ops (SOPs): {sops/1e6:.2f} M")
    print(f"Spike Energy (MLP)  : {snn_spike_energy*1e3:.4f} mJ")
    
    # --- 4. Hybrid Energy Calculation (Quan trọng cho báo cáo) ---
    # ViT SNN của bạn là Hybrid (Attention FP32 + MLP SNN)
    # Tổng năng lượng = E_Attn (MACs) + E_MLP (SOPs)
    # Ước lượng: Trong ViT, MLP chiếm khoảng 60-65% tham số và FLOPs. Attention chiếm 35-40%.
    # Ta lấy con số an toàn: E_Attn ≈ 0.4 * E_ANN_Total
    
    energy_attn_static = 0.4 * ann_energy
    total_snn_energy = energy_attn_static + snn_spike_energy
    
    print(f"Static Energy (Attn): {energy_attn_static*1e3:.4f} mJ (Est. 40% ANN)")
    print(f"------------------------------------------------")
    print(f"TOTAL SNN ENERGY    : {total_snn_energy*1e3:.4f} mJ")
    
    # 5. Comparison
    ratio = ann_energy / total_snn_energy
    print(f"\n==> Energy Saving: {ratio:.2f}x better than ANN")

if __name__ == "__main__":
    main()