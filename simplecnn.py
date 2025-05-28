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

import FGSM_Perturbed_Images_Facenet

# Data Preparation

# Random Seed for Reproducibility
torch.manual_seed(42)
np.random.seed(42)

def load_data(data_dir, height, length): 
 

    # Formatting the Dataset

    transform = transforms.Compose([
        transforms.Resize((height, length)),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5)) # normalizes from [0,1] --> [-1,1]
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform = transform)

    # Train/Validation/Test Split
    total_size = len(dataset)  # 70/15/15 Trin/Validation/Test split
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    return train_set, val_set, test_set

# Displaying an Image

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


#Model Definition


class SimpleCNN(nn.Module):
    def __init__(self,num_classes, height, length):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 3, padding = 1, stride = 1)

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 3, padding = 1, stride = 1)

        self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1, stride = 1)

        h_out = height // 2 // 2 // 2
        l_out = length // 2 // 2 // 2
        flat_feats = 32 * h_out * l_out


        self.fc1 = nn.Linear(flat_feats, 120)
        self.fc2 = nn.Linear(120,        84)
        self.fc3 = nn.Linear(84,         20)
        self.fc4 = nn.Linear(20,  num_classes)

        self.dropout = nn.Dropout(p = 0.25)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)


        return x

# Criterion

criterion = nn.CrossEntropyLoss() #Cross Entropy (MLE assuming Categorical Distribution)

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



def train_model(batch_size, epochs, lr, weight_decay, n_classes, train_set, val_set, height, length):
    val_loader = DataLoader(val_set, batch_size = 512)
    val_accuracy = 0


    model = SimpleCNN(n_classes, height, length)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True)

    for epoch in range(1, epochs + 1):
      for images, labels in train_loader:

          optimizer.zero_grad()

          outputs = model.forward(images)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          print("Loss:", round(loss.item(), 3))

      val_accuracy = compute_accuracy(model, val_loader)
      train_accuracy = compute_accuracy(model, train_loader)

      print("Epoch:", epoch, "Validation Accuracy:", round(val_accuracy, 3), '%', "Training Accuracy: ", round(train_accuracy, 3) )

    return model

def build_model(train_set, val_set, height, length):
    batch_size = 128
    epoch = 20
    lr = 0.001
    weight_decay = 0.001
    classes = train_set.dataset.classes
    n_classes = len(classes)
    model = train_model(batch_size, epoch, lr, weight_decay, n_classes, train_set, val_set, height, length)
    return model


def save_model(model, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model, model_path)

def load_model(model_path): 
    model = torch.load(model_path, weights_only = False)
    return model


if __name__ == "__main__":
#Directory
    data_dir = "just_faces"
    img_height = 160  #resize all images to uniform size
    img_length = 160

#Get the datasets
    train_set, val_set, test_set = load_data(data_dir, img_height, img_length)

#Build the Model
    # model = build_model(train_set, val_set, img_height, img_length)

#Save the Model
    model_path = "models/simplecnn.pth"
    # save_model(model, model_path)

#Load Model
    model = load_model(model_path)
    















