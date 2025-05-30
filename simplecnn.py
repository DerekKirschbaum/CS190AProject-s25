import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from data import classes, HEIGHT, LENGTH, VAL_SET

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
    
     # Model Training

    def build(self, dataset, save_path, batch_size = 128, epochs = 5, lr = 0.001, weight_decay = 0.001, val_set = VAL_SET, is_verbose = False):
        print("Building SimpleCNN...")
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
        print("SimpleCNN Build Complete")
    
    def compute_gradient(self, image, celebrity):  #image: Tensor [3,160,160], celebrity: string, e.g. "Tom Hanks"
        self.eval()
        
        image = image.clone().unsqueeze(0).requires_grad_(True)
        
        idx = classes.index(celebrity)
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
   

    def save(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(self.state_dict(), file_path)

    def load(self, file_path): 
        self.load_state_dict(torch.load(file_path))




    
    



    


