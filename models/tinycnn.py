import torch
import torch.nn as nn
import torch.nn.functional as F
from models.homemade_classifier import Classifier

from preprocess_data import VAL_SET, CLASSES

class TinyCNN(Classifier):
    def __init__(self,
                 classes=CLASSES,
                 height=160,
                 length=160,
                 model_name="SimpleCNN"):
        self.classes = classes
        self.height = height
        self.length = length

        super().__init__(model_name=model_name)

        # One 3×3 conv layer: in_channels=3 → out_channels=8
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            padding=1,
            stride=1
        )

        # One 2×2 max‐pool
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After conv + pool, spatial dims: 160 -> 80, then flatten
        h_out = self.height // 2
        l_out = self.length // 2
        flat_feats = 8 * h_out * l_out

        # final fully‐connected layer
        self.fc = nn.Linear(flat_feats, len(self.classes))

        # compute cross‐entropy loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # x: [B, 3, 160, 160]
        x = F.relu(self.conv1(x))    # [B, 8, 160, 160]
        x = self.pool(x)             # [B, 8,  80,  80]
        x = torch.flatten(x, 1)      # [B, 8*80*80]
        x = self.fc(x)               # [B, num_classes]
        return x

    



    
    



    


