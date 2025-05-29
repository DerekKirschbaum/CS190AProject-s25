# Imports
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

# Data Preparation

#Constants
HEIGHT = 160
LENGTH = 160
CRITERION = nn.CrossEntropyLoss() #Cross Entropy (MLE assuming Categorical Distribution)
DATA_DIR = './just_faces'
MODEL_PATH = './models/simplecnn.pth'


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


#Model Definition


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 3, padding = 1, stride = 1)

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 3, padding = 1, stride = 1)

        h_out = HEIGHT // 2 // 2 
        l_out = LENGTH // 2 // 2
        flat_feats = 16 * h_out * l_out


        self.fc1 = nn.Linear(flat_feats, 120)
        self.fc2 = nn.Linear(120,        84)
        self.fc3 = nn.Linear(84,         20)
        self.fc4 = nn.Linear(20,  len(classes))

        self.dropout = nn.Dropout(p = 0.25)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))


        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)


        return x


# Computing Accuracy

def compute_accuracy(model, data_loader):
  correct = 0
  total = 0

  model.eval()
  with torch.no_grad():
      for images, labels in data_loader:
          outputs = model(images)

          _, predicted = torch.max(outputs, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  accuracy = correct / total * 100
  return accuracy

# Model Training


def train_model(batch_size, epochs, lr, weight_decay):
    val_loader = DataLoader(val_set, batch_size = 512)
    val_accuracy = 0

    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay) 
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
      for images, labels in train_loader:

          optimizer.zero_grad()

          outputs = model.forward(images)
          loss = CRITERION(outputs, labels)
          loss.backward()
          optimizer.step()
          print("Loss:", round(loss.item(), 3))

      val_accuracy = compute_accuracy(model, val_loader)
      train_accuracy = compute_accuracy(model, train_loader)

      print("Epoch:", epoch, "Validation Accuracy:", round(val_accuracy, 3), '%', "Training Accuracy: ", round(train_accuracy, 3) )

    return model


def save_model(model):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model, MODEL_PATH)

def load_model(): 
    model = torch.load(MODEL_PATH, weights_only = False)
    return model


if __name__ == "__main__":
    model = train_model(batch_size = 128, epochs = 5, lr = 0.001, weight_decay = 0.001)
    save_model(model)
    



    


