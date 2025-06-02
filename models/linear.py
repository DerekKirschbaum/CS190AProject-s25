import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from models.homemade_classifier import Classifier

from preprocess_data import CLASSES

#Model Definition

class SimpleCNN(Classifier):
    def __init__(self, classes = CLASSES, height = 160, length = 160):
        self.classes = classes
        self.height = height
        self.length = length

        super().__init__()

        flat_feats = 3 * height * length


        self.fc1 = nn.Linear(flat_feats, 120)
        self.fc2 = nn.Linear(120,        84)
        self.fc3 = nn.Linear(84,         20)
        self.fc4 = nn.Linear(20,  len(self.classes))

        self.dropout = nn.Dropout(p = 0)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x
    



    
    



    


