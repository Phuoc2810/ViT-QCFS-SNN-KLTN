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


def get_dataloader(batch_size=Config.BATCH_SIZE, dataset_name='cifar10'):
    """
    Hàm tạo dataloader linh hoạt cho cả CIFAR-10 và CIFAR-100.
    
    Args:
        batch_size (int): Kích thước batch.
        dataset_name (str): Tên dataset ('cifar10' hoặc 'cifar100').
        
    Returns:
        train_loader, test_loader, num_classes
    """
    
    # Chuẩn hóa tên dataset để tránh lỗi viết hoa/thường
    dataset_name = dataset_name.lower()
    print(f"==> Preparing data for {dataset_name.upper()}...")

    # 1. Cấu hình Mean/Std riêng cho từng bộ dữ liệu
    # CIFAR-100 có mean/std hơi khác CIFAR-10 một chút
    if dataset_name == 'cifar100':
        mean = Config.CIFAR100_MEAN
        std = Config.CIFAR100_STD
        num_classes = 100
    else:
        # Mặc định là CIFAR-10
        mean = Config.CIFAR_MEAN
        std = Config.CIFAR_STD
        num_classes = 10

    # 2. Định nghĩa Transform (Dùng chung logic Augmentation)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(), # Augmentation hiện đại
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # 3. Tải Dataset
    if dataset_name == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test
        )
    else:
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )

    # 4. Tạo DataLoader
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

    print(f"✅ Data loaded: {dataset_name.upper()} | Classes: {num_classes} | Train batches: {len(train_loader)}")
    
    # Quan trọng: Trả về cả num_classes để Model tự chỉnh đầu ra
    return train_loader, test_loader, num_classes