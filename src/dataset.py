import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms 
from tqdm import tqdm 
from src.config import Config

# # --- Data Augmentation and Normalization ---

# # Transformations for the training set - THIS IS THE CRITICAL FIX
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
    
#     # --- ADDED: Strong, modern augmentation ---
#     # Applies a sequence of randomly selected and configured augmentations.
#     # This is a standard, highly effective technique for ViTs.
#     transforms.TrivialAugmentWide(),
    
#     transforms.ToTensor(), # Convert image to a PyTorch Tensor
    
#     # --- ADDED: Cutout / Random Erasing ---
#     # Randomly erases a rectangular region in the image. This prevents
#     # the model from focusing too much on any one specific feature.
#     # It forces the model to learn more distributed, robust features.
#     transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    
#     transforms.Normalize(Config.CIFAR_MEAN, Config.CIFAR_STD), # Normalize
# ])

# # The test transform remains unchanged. We never augment validation/test data.
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(Config.CIFAR_MEAN, Config.CIFAR_STD),
# ])

# # --- Create Datasets and DataLoaders ---

# # Download the datasets

# #--- Cifar 10 ----
# train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# #--- Cifar 100 ----
# # train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
# # test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

# # Create the DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
# test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# print(f"Data loaded. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")


def get_dataloader(batch_size=Config.BATCH_SIZE):
    """
    Hàm tạo và trả về train_loader, test_loader.
    
    Args:
        batch_size (int): Kích thước batch, mặc định lấy từ Config.
                          Cho phép ghi đè từ dòng lệnh (argparse).
    Returns:
        train_loader, test_loader
    """
    
    # --- Data Augmentation and Normalization ---
    # Transformations for the training set
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        
        # --- ADDED: Strong, modern augmentation ---
        transforms.TrivialAugmentWide(),
        
        transforms.ToTensor(), # Convert image to a PyTorch Tensor
        
        # --- ADDED: Cutout / Random Erasing ---
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        
        transforms.Normalize(Config.CIFAR_MEAN, Config.CIFAR_STD), # Normalize
    ])

    # The test transform remains unchanged.
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(Config.CIFAR_MEAN, Config.CIFAR_STD),
    ])

    # --- Create Datasets ---
    # root='./data' giúp gom dữ liệu vào folder data gọn gàng
    # download=True sẽ tự động tải nếu chưa có
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    # --- Create DataLoaders ---
    # Lưu ý: Trên Windows, nếu num_workers > 0 có thể gây lỗi hoặc chậm khi khởi động.
    # Nếu chạy bị treo, hãy thử set num_workers=0.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )

    print(f"Data loaded. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    return train_loader, test_loader