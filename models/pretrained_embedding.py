# base_embedding_model.py
import numpy as np
from abc import ABC, abstractmethod
from torchvision import transforms
from typing import Dict
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from preprocess_data import CLASSES


class EmbeddingModel(ABC): #ABC = abstract base class
    def __init__(self, model, model_name):
        self.class_means: Dict[str, np.ndarray] = {}  
        self.model = model
        self.to_tensor = transforms.ToTensor()
        self.model_name = model_name
        self.classes = None
        self.mtcnn = MTCNN(image_size=160, margin=0)

    def build(self, dataset, save_path: str):
        
        self.classes = dataset.dataset.classes
        print(f"Building {self.model_name}…")
        embeddings_by_class = {name: [] for name in self.classes}

        for img_tensor, label_idx in dataset:
            # img_tensor = [3,H,W]
            # embed() expects a 4D batch.
            single_batch = img_tensor.unsqueeze(0)       # convert to [1,3,H,W]
            emb_tensor = self.embed(single_batch)       
            emb_np = emb_tensor.cpu().numpy()[0]         
            class_name = self.classes[label_idx]
            embeddings_by_class[class_name].append(emb_np)

        # store mean‐vector per class
        for class_name, embs in embeddings_by_class.items():
            mean_emb = np.mean(embs, axis=0)
            self.class_means[class_name] = mean_emb / np.linalg.norm(mean_emb)

        print(f"{self.model_name} Build Complete")

        self.save(save_path)

    def forward(self, x):
        #get embedding of x
        emb = self.embed(x)           
        emb = F.normalize(emb, dim=1)

        cm_torch = torch.stack(
            [torch.from_numpy(m).to(x.device).float()
             for m in self.class_means.values()],
            dim=1
        )  # [D, num_classes]
        return emb @ cm_torch
 
    def embed(self, face_tensor: torch.Tensor) -> np.ndarray:
        if face_tensor.dim() == 3:
            x_in = face_tensor.unsqueeze(0)   # -> [1, 3, H, W]
        elif face_tensor.dim() == 4:
            x_in = face_tensor            
        else:
            raise ValueError(
                f"embed(...) expected input of dim 3 or 4, got {face_tensor.dim()}"
            )

        emb = self.model(x_in)                
        emb = F.normalize(emb, p=2, dim=1)       

        return emb   
    
    def compute_gradient(self, image, celebrity):  # tensor [3,160,160], celebrity: string
        cm_torch = torch.stack([torch.tensor(self.class_means[c], dtype=torch.float32)
                                for c in self.classes], dim=1)
        x = image.unsqueeze(0).clone().detach().requires_grad_(True) # (1,3,160,160)
        logits = self.forward(x)

        y_true = self.classes.index(celebrity)
        label = torch.tensor([y_true], dtype=torch.long)

        # compute cross entropy loss
        loss = F.cross_entropy(logits, label)
        loss.backward()

        grad = x.grad   
        grad = grad.squeeze(dim = 0) # (3,160,160)

        return grad

    def compute_accuracy(self, dataset):
        correct = 0
        total = 0

        with torch.no_grad():
            for image, label in dataset:
                x = image.unsqueeze(0)
                logits = self.forward(x)

                pred_idx = int(logits.argmax(dim=1).item())
                true_idx = int(label)

                if pred_idx == true_idx:
                    correct += 1

                total += 1

        return (correct / total) * 100.0 if total > 0 else 0.0
    def save(self, file_path: str):
        checkpoint = {
            'class_means': self.class_means,
            'classes':     self.classes
        }
        np.save(file_path, checkpoint, allow_pickle=True)

    def load(self, file_path: str):
        loaded = np.load(file_path, allow_pickle=True).item()
        self.class_means  =  loaded['class_means']   
        self.classes = loaded['classes']    

    def compute_accuracy_with_cos(self, dataset, threshold): 
        correct = 0
        cos = 0
        total = 0
        for image, label in dataset: 
            image = image.unsqueeze(0)  # Add batch dimension: [1, C, H, W]
            cos_sim = self.forward(image)
            
            # Get predicted class and similarity value
            val, pred_idx = torch.max(cos_sim, dim=1)
            pred_idx = pred_idx.item()
            val = val.item()
            
            if pred_idx == label:
                correct += 1
                if val >= threshold:
                    cos += 1

            total += 1
        accreg = (correct / total) * 100
        acccos = (cos / total) * 100
        return accreg, acccos