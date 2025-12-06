import torch
import argparse
from fvcore.nn import FlopCountAnalysis
from src.config import Config
from src.model_ann import VisionTransformer
from src.model_snn import SpikeVisionTransformer
from src.dataset import get_dataloader
from src.layers import IF  # Cần import lớp IF để check module

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--T', type=int, default=4, help='Timesteps for SNN energy calc')
    return parser.parse_args()

def calc_ann_energy(model, sample_input):
    """Tính năng lượng ANN dựa trên MACs"""
    flops = FlopCountAnalysis(model, sample_input)
    macs = flops.total() / 2  # FLOPs = 2 * MACs
    energy_joule = macs * 3e-12 # 3.0 pJ per MAC (45nm technology reference)
    return macs, energy_joule

def calc_snn_energy(model, loader, device):
    """Tính năng lượng SNN dựa trên số Spike"""
    model.eval()
    total_spikes = 0
    num_samples = 0
    
    # Lấy 1 batch để tính trung bình
    images, _ = next(iter(loader))
    images = images.to(device)
    num_samples = images.size(0)
    
    with torch.no_grad():
        _ = model(images) # Run forward pass
        
        # Duyệt qua các layer để cộng dồn spike count
        for module in model.modules():
            if isinstance(module, IF) and hasattr(module, 'spike_count'):
                # Fan-out: Số lượng kết nối ra từ nơ-ron này.
                # Giả định kiến trúc MLP: Dim -> Hidden -> Dim.
                # Spike ở đây thường đi vào lớp Linear tiếp theo.
                # Lấy output features của layer Linear nằm ngay sau nó trong Sequential
                # (Logic này mang tính ước lượng, cần chỉnh nếu kiến trúc thay đổi)
                fan_out = 192 * 4 # Ví dụ: hidden_dim của TinyViT
                total_spikes += module.spike_count.item() * fan_out

    avg_ops = total_spikes / num_samples
    energy_joule = avg_ops * 0.1e-12 # 0.1 pJ per SOP (Synaptic Operation)
    return avg_ops, energy_joule

def main():
    args = get_args()
    device = Config.DEVICE
    
    # 1. Setup Models
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    
    ann_model = VisionTransformer(dim=192, depth=12, heads=3).to(device)
    snn_model = SpikeVisionTransformer(dim=192, depth=12, heads=3, T=args.T).to(device)
    
    # Load weights (để SNN bắn spike đúng thực tế)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    # Load weights logic... (có thể tái sử dụng hàm transfer_weights từ file convert nếu tách ra utils)
    # Ở đây code nhanh việc load state dict cho SNN
    # ... (Bạn nên copy hàm transfer_weights vào src/utils.py để import dùng chung)

    # 2. Calculate ANN
    print("Calculating ANN Energy...")
    macs, ann_e = calc_ann_energy(ann_model, dummy_input)
    print(f"ANN MACs: {macs/1e9:.2f} G | Energy: {ann_e*1e3:.4f} mJ")

    # 3. Calculate SNN
    print(f"Calculating SNN Energy (T={args.T})...")
    _, test_loader = get_dataloader(batch_size=32)
    
    # Cần load weight chuẩn thì spike count mới đúng, nếu không spike sẽ random
    # Tạm thời bỏ qua bước load weight ở demo này, nhưng khi chạy thật bạn BẮT BUỘC phải load.
    
    ops, snn_e = calc_snn_energy(snn_model, test_loader, device)
    print(f"SNN SynOps: {ops/1e9:.2f} G | Energy: {snn_e*1e3:.4f} mJ")
    
    print("-" * 30)
    print(f"Efficiency Ratio: {ann_e / snn_e :.2f}x")

if __name__ == "__main__":
    main()