import os
import random
import numpy as np
import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms 
from tqdm import tqdm 
from src.config import Config
from src.layers import *

# File: scripts/train_ann.py hoặc src/utils.py
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train ANN Vision Transformer')

    # Các tham số cơ bản
    parser.add_argument('--batch_size', type=int, default=128, help='Kích thước batch')
    parser.add_argument('--epochs', type=int, default=200, help='Số lượng epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Các tham số về Model (ViT)
    parser.add_argument('--patch_size', type=int, default=4, help='Kích thước patch ảnh')
    parser.add_argument('--dim', type=int, default=192, help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=12, help='Độ sâu của Transformer (số layers)')
    parser.add_argument('--heads', type=int, default=3, help='Số lượng Attention Heads')
    
    # Các tham số về đường dẫn (Path)
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Nơi lưu model')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='Đường dẫn file .pth để train tiếp (nếu có)')
    parser.add_argument('--no_progress_bar', action='store_true', help='Tắt thanh tiến trình (tqdm) nếu chạy ngầm')

    args = parser.parse_args()
    return args

def set_seed(seed):
    """ Sets the random seed for the entire project for reporducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # set seed for the current gpu
        torch.cuda.manual_seed_all(seed) # set seed for all the gpus available later on
        
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Set the seed for our entire run
set_seed(Config.SEED)
print(f"Random seed set to: {Config.SEED}")


def calculate_snn_synops(model, dataloader, device):
    """
    Tính toán số lượng phép toán synapse (SynOps) cho SNN.
    Args:
        model: Mô hình SNN cần tính.
        dataloader: Dữ liệu dùng để chạy test (thường là test_loader).
        device: 'cuda' hoặc 'cpu'.
    """
    model.eval()
    total_synops = 0
    num_samples = 0

    with torch.no_grad():
        # Lấy 1 batch mẫu từ dataloader được truyền vào
        images, _ = next(iter(dataloader))
        images = images.to(device)
        num_samples = images.shape[0]

        # Chạy forward để kích hoạt spike_count trong các lớp IF
        _ = model(images)

        # Duyệt qua các module để cộng dồn spike
        for module in model.modules():
            # Kiểm tra xem module có phải là lớp IF không
            if isinstance(module, IF) and module.T > 0:
                # Tính fan_out (số nơ-ron lớp sau). 
                # Lưu ý: Logic này phụ thuộc vào kiến trúc cụ thể của bạn (TinyViT)
                # dim=192, mlp_ratio=4 => hidden_dim = 768
                fan_out = 192 * 4 
                
                # Sửa snn_model -> module (vì đang duyệt loop)
                # Sửa spike_count thành item() để lấy giá trị số
                layer_synops = module.spike_count.item() * fan_out
                total_synops += layer_synops

    avg_synops_per_image = total_synops / num_samples
    return avg_synops_per_image


def prune_model(model, amount= 0.2):
    """
    Cắt tỉa trọng số của mô hình (L1 Unstructured Pruning).
    
    Args:
        model: Mô hình ANN cần cắt tỉa.
        amount (float): Tỷ lệ cắt tỉa (0.0 đến 1.0). Ví dụ 0.2 là cắt 20% weight bé nhất.
    """
    print(f"✂️ Đang tiến hành Pruning (Cắt tỉa) với tỷ lệ: {amount*100}% ...")
    
    parameters_to_prune = []
    
    # 1. Thu thập tất cả các lớp Linear trong model
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    # 2. Thực hiện cắt tỉa toàn cục (Global Pruning)
    # Cơ chế: Gom tất cả weight lại, tìm 20% bé nhất trên toàn mạng và gán = 0
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    # 3. "Làm cứng" việc cắt tỉa (Make permanent)
    # PyTorch mặc định chỉ tạo mặt nạ (mask), ta cần lệnh này để weight thực sự biến thành 0
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')
        
    # 4. Kiểm tra tỷ lệ zero thực tế (để confirm)
    total_zeros = 0
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_zeros += torch.sum(module.weight == 0).item()
            total_params += module.weight.nelement()
            
    print(f"✅ Pruning hoàn tất! Tỷ lệ thưa (Sparsity): {100. * total_zeros / total_params:.2f}%")
    return model
# --- Usage ---
# You need your trained SNN model and the test_loader
# snn_model = SpikeVisionTransformer(...)
# snn_model.load_state_dict(...)
# snn_model.to(Config.DEVICE)
