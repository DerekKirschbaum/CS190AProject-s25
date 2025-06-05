import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from preprocess_data import VAL_SET, CLASSES
from abc import ABC, abstractmethod

#Model Definition

class Classifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.model_name = model_name

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

     # Model Training

    def build(self, dataset, save_path, batch_size = 128, epochs = 10, lr = 0.001, weight_decay = 0.001, val_set = VAL_SET, is_verbose = False):
        print("Building " + self.model_name + "...")
        max_validation_accuracy = 0
        val_accuracy = 0

        optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = weight_decay) 
        
        train_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle=True)

        for epoch in range(1, epochs + 1):
            for images, labels in train_loader:

                optimizer.zero_grad()

                outputs = self.forward(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if(is_verbose): 
                    print("Loss:", round(loss.item(), 3))

            val_accuracy = self.compute_accuracy(val_set)
            train_accuracy = self.compute_accuracy(dataset)

            if(val_accuracy > max_validation_accuracy): 
                max_validation_accuracy = val_accuracy
                self.save(save_path)

            if(is_verbose): 
                print("Epoch:", epoch, "Validation Accuracy:", round(val_accuracy, 3), '%', "Training Accuracy: ", round(train_accuracy, 3) )
        print(self.model_name + " Build Complete")
    
    def compute_gradient(self, image, celebrity):  #image: Tensor [3,160,160], celebrity: string, e.g. "Tom Hanks"
        self.eval()
        
        image = image.clone().unsqueeze(0).requires_grad_(True)
        
        idx = CLASSES.index(celebrity)
        label = torch.tensor([idx], dtype=torch.long)

        output = self.forward(image)  
        loss   = self.criterion(output, label)
        loss.backward()
        
        grad = image.grad.detach().squeeze(0)  # [3,160,160]
        grad = grad.clamp(-1,1)
        return grad

    def compute_accuracy(self, dataset): #Computing Accuracy
        correct = 0
        total = 0
        self.eval()
        for image, label in dataset:
            image = image.unsqueeze(dim = 0)
            outputs = self.forward(image)
            _, predicted = torch.max(outputs, 1)
            if(label == predicted): 
                correct += 1
            total += 1
        accuracy = correct / total * 100
        return accuracy
    
    def compute_accuracy_with_cos(self, dataset, threshold = 0.5):
        correct = 0
        cos = 0
        total = 0
        self.eval()
        softmax = nn.Softmax(dim=1)
        for image, label in dataset:
            image = image.unsqueeze(0)
            outputs = self.forward(image)
            probs = softmax(outputs)
            max_prob, predicted = torch.max(probs, dim=1)
            _, pred = torch.max(outputs, 1)
            if(label == pred): 
                correct += 1
            if predicted.item() == label and max_prob.item() >= threshold:
                cos += 1
            total += 1
        acc_reg = (correct / total) * 100
        acc_cos = (cos / total) * 100
        return acc_reg, acc_cos
   

    def save(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(self.state_dict(), file_path)

    def load(self, file_path): 
        self.load_state_dict(torch.load(file_path))




    
    



    


