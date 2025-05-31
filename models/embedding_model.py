# base_embedding_model.py
import numpy as np
from abc import ABC, abstractmethod
from torchvision import transforms
from typing import Dict
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN


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
        print("Building " + self.model_name + "...")
        embeddings_by_class = {name: [] for name in self.classes}
        for img_tensor, label in dataset:
            name = self.classes[label]
            emb = self.embed(img_tensor)
            embeddings_by_class[name].append(emb)
        self.class_means = {}
        for name, embs in embeddings_by_class.items():
            mean_emb = np.mean(embs, axis=0)
            self.class_means[name] = mean_emb / np.linalg.norm(mean_emb)
        self.save(save_path)
        print(self.model_name + " Build Complete")

    def forward(self, x):
        class_means = self.class_means
        emb = self.embed(x)
        emb = emb / np.linalg.norm(emb)
        sims = {c: np.dot(emb, class_means[c]) for c in self.classes}
        pred = max(sims, key=sims.get)
        return pred
 
    def embed(self, face_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            emb = self.model(face_tensor.unsqueeze(0))
            emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()[0]

    
    def compute_gradient(self, image, celebrity):  # tensor [3,160,160], celebrity = 'scarlett', 'brad',
        # 1) build a stacked tensor of all class‐means: shape (512,5)
        cm_torch = torch.stack([torch.tensor(self.class_means[c], dtype=torch.float32)
                                for c in self.classes], dim=1)
        # 2) prepare input for gradient
        x = image.unsqueeze(0).clone().detach().requires_grad_(True) #(1,3,160,160)
        # 3) forward → embedding → normalize → compute 5 “logits”
        emb = self.model(x)                                          # (1,512)
        embn = F.normalize(emb, p=2, dim=1)                     # (1,512)
        logits = embn @ cm_torch                                # (1,5)

        y_true = self.classes.index(celebrity)
        label = torch.tensor([y_true], dtype=torch.long)

        # 5) cross-entropy loss (we want to *increase* this)
        loss = F.cross_entropy(logits, label)
        loss.backward()

        grad = x.grad   
        grad = grad.squeeze(dim = 0)                        # (3,160,160)

        return grad

    def compute_accuracy(self, dataset):
        correct = 0
        total = 0
        for image, label in dataset:
            true_name = self.classes[label]
            pred_name = self.forward(image)
            if pred_name == true_name:
                correct += 1
            total += 1

        return (correct / total) * 100 if total > 0 else 0.0

    def save(self, file_path: str):
        checkpoint = {
            'class_means': self.class_means,
            'classes':     self.classes
        }
        np.save(file_path, checkpoint, allow_pickle=True)

    def load(self, file_path: str):
        loaded = np.load(file_path, allow_pickle=True).item()
        self.class_means   = loaded['class_means']   
        self.classes  = loaded['classes']    
