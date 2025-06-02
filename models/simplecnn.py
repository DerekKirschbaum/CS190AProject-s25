import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from models.homemade_classifier import Classifier

from preprocess_data import VAL_SET, CLASSES

#Model Definition

class SimpleCNN(Classifier):
    def __init__(self, classes = CLASSES, height = 160, length = 160):
        self.classes = classes
        self.height = height
        self.length = length

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 3, padding = 1, stride = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 3, padding = 1, stride = 1)

        h_out = self.height // 2 // 2 
        l_out = self.height // 2 // 2
        flat_feats = 16 * h_out * l_out

        self.fc1 = nn.Linear(flat_feats, 120)
        self.fc2 = nn.Linear(120,        84)
        self.fc3 = nn.Linear(84,         20)
        self.fc4 = nn.Linear(20,  len(self.classes))

        self.dropout = nn.Dropout(p = 0)
        self.criterion = nn.CrossEntropyLoss()

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
    



    
    



    


