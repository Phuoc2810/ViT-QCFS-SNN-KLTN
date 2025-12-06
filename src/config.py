import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms 
from tqdm import tqdm 

class Config:
    # --- Project Control ---
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Training Hyperparameters ---
    EPOCHS = 250
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3

    # --- Dataset Information
    # They are used to normalize the images to a similar scale

    #--- Cifar 10 ---
    CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR_STD = (0.2023, 0.1994, 0.2010)

    #--- Cifar 100 ---
    # CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
    # CIFAR_STD  = (0.2675, 0.2565, 0.2761)


    # --- File Paths ---
    CHECKPOINT_DIR = "./checkpoints/"
    
print(f"Device set to: {Config.DEVICE}")

class EvalConfig:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 128
    CIFAR_MEAN = Config.CIFAR_MEAN
    CIFAR_STD  = Config.CIFAR_STD