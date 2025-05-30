import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import os


HEIGHT = 160
LENGTH = 160 
DATA_DIR = './Dataset'


# Random Seed for Reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Formatting the Dataset

transform = transforms.Compose([
    transforms.Resize((HEIGHT, LENGTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5)) # normalizes from [0,1] --> [-1,1]
])

dataset = datasets.ImageFolder(root = DATA_DIR, transform = transform)
classes = dataset.classes

# Train/Validation/Test Split
total_size = len(dataset)  # 70/15/15 Trin/Validation/Test split
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])