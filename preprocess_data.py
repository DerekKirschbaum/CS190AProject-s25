import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import random_split

HEIGHT = 160
LENGTH = 160 
DATA_DIR = './Dataset'

# Manually set random seed
torch.manual_seed(42)
np.random.seed(42)

transform = transforms.Compose([
    transforms.Resize((HEIGHT, LENGTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5)) # normalizes from [0,1] to [-1,1]
])

DATASET = datasets.ImageFolder(root = DATA_DIR, transform = transform)
CLASSES = DATASET.classes

# 70/15/15 Train/Validation/Test Split
total_size = len(DATASET) 
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

TRAIN_SET, VAL_SET, TEST_SET = random_split(DATASET, [train_size, val_size, test_size])