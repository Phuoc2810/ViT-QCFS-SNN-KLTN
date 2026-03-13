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
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to checkpoint")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--T', type=int, default=4, help='Timesteps')
    parser.add_argument('--embed_dim', type=int, default=192)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--heads', type=int, default=3)
    return parser.parse_args()

def calc_ann_flops(model, input_tensor):
    model.eval()
    flops = FlopCountAnalysis(model, input_tensor)
    total_flops = flops.total()
    macs = total_flops / 2 
    energy_joule = macs * 4.6e-12 
    return macs, energy_joule

def calc_snn_energy(model, loader, device, T, embed_dim):
    """
    Tính năng lượng Dynamic Spike (SOPs) trên toàn bộ tập test.
    """
    model.eval()
    # Reset spike count
    for m in model.modules():
        if hasattr(m, 'spike_count'):
            m.spike_count.zero_()

    print(f"Running simulation to count spikes (T={T})...")
    total_samples = 0
    
    # Chạy max 50 batch để tiết kiệm thời gian mà vẫn đủ chính xác
    max_batches = 50 
    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(loader, total=max_batches)):
            if i >= max_batches: break
            images = images.to(device)
            _ = model(images)
            total_samples += images.size(0)

    # Tính SOPs
    sops = 0
    for name, module in model.named_modules():
        if isinstance(module, IF) and hasattr(module, 'spike_count'):
            fan_out = embed_dim 
            avg_spikes = module.spike_count.item() / total_samples
            sops += avg_spikes * fan_out

    # Energy per SOP (0.9pJ)
    energy_spike = sops * 0.9e-12 
    return sops, energy_spike

def main():
    args = get_args()
    device = Config.DEVICE
    
    # 1. Load Checkpoint & Config
    print(f"Loading checkpoint: {args.checkpoint_path}")
    # Xử lý load checkpoint an toàn
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
    except:
        checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)

    if args.dataset == 'cifar100': num_classes = 100
    else: num_classes = 10

    # Lấy config từ checkpoint nếu có
    saved_args = checkpoint.get('args', None)
    if saved_args:
        embed_dim = saved_args.embed_dim
        depth = saved_args.depth
        heads = saved_args.heads
    else:
        embed_dim = args.embed_dim
        depth = args.depth
        heads = args.heads

    print(f"Config: Dim={embed_dim}, Depth={depth}, Heads={heads}, T={args.T}")

    # 2. ANN Baseline Analysis
    ann_model = VisionTransformer(dim=embed_dim, depth=depth, heads=heads, num_classes=num_classes, T=0).to(device)
    dummy_input = torch.randn(1, 3, 32, 32).to(device)

    print("\n--- 1. ANN Baseline Analysis ---")
    macs, ann_energy = calc_ann_flops(ann_model, dummy_input)
    print(f"Total MACs (FP32)    : {macs/1e6:.2f} M")
    print(f"Total ANN Energy     : {ann_energy*1e3:.4f} mJ")

    # 3. SNN Analysis
    snn_model = SpikeVisionTransformer(dim=embed_dim, depth=depth, heads=heads, num_classes=num_classes, T=args.T).to(device)
    
    # Load weights
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    snn_model.load_state_dict(state_dict, strict=False)

    _, test_loader, _ = get_dataloader(batch_size=32, dataset_name=args.dataset)

    print(f"\n--- 2. SNN Dynamic Component (Spikes) ---")
    sops, snn_spike_energy = calc_snn_energy(snn_model, test_loader, device, args.T, embed_dim)
    print(f"Avg Spike Ops (SOPs) : {sops/1e6:.2f} M")
    print(f"Dynamic Energy (MLP) : {snn_spike_energy*1e3:.4f} mJ")

    # --- PHẦN QUAN TRỌNG: 2 SCENARIOS ---
    # Ước lượng: Attention Block chiếm 40% FLOPs của ANN gốc
    static_energy_unit = 0.4 * ann_energy 

    print(f"\n==============================================")
    print(f"   ENERGY COMPARISON REPORT (T={args.T})")
    print(f"==============================================")

    # SCENARIO A: Ideal / Hardware Optimized
    # Giả định: Attention chỉ tính 1 lần và broadcast (One-shot Attention)
    total_A = static_energy_unit + snn_spike_energy
    ratio_A = ann_energy / total_A
    
    print(f"\n[Scenario A] Ideal Optimization (Static calculated 1 time):")
    print(f" - Static (Attn) : {static_energy_unit*1e3:.4f} mJ")
    print(f" - Dynamic (MLP) : {snn_spike_energy*1e3:.4f} mJ")
    print(f" -> TOTAL SNN    : {total_A*1e3:.4f} mJ")
    print(f" -> SAVING       : {ratio_A:.2f}x vs ANN")

    # SCENARIO B: Realistic / Current Implementation
    # Thực tế: Attention phải chạy T lần vì Tensor Input là (T*B, N, D)
    static_energy_T = static_energy_unit * args.T
    total_B = static_energy_T + snn_spike_energy
    
    print(f"\n[Scenario B] Current Implementation (Static calculated {args.T} times):")
    print(f" - Static (Attn) : {static_energy_T*1e3:.4f} mJ")
    print(f" - Dynamic (MLP) : {snn_spike_energy*1e3:.4f} mJ")
    print(f" -> TOTAL SNN    : {total_B*1e3:.4f} mJ")
    
    if total_B < ann_energy:
        ratio_B = ann_energy / total_B
        print(f" -> SAVING       : {ratio_B:.2f}x vs ANN")
    else:
        ratio_B = total_B / ann_energy
        print(f" -> COST         : {ratio_B:.2f}x MORE than ANN (Latency Trade-off)")

if __name__ == "__main__":
    main()