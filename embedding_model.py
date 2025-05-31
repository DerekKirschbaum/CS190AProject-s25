# base_embedding_model.py

import os
import numpy as np
from abc import ABC, abstractmethod
from torchvision import transforms
from typing import Dict


class EmbeddingModel(ABC): #ABC = abstract base class
    def __init__(self, model, model_name):
        self.class_means: Dict[str, np.ndarray] = {}  
        self.model = model
        self.to_tensor = transforms.ToTensor()
        self.model_name = model_name
        self.classes

    @abstractmethod
    def build(self, dataset, save_path: str):
        self.classes = dataset.dataset.classes
        print("Building " + self.model_name)
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
 
    @abstractmethod
    def embed(self, x):
        pass
    
    @abstractmethod
    def compute_gradient(self, x): 
        pass

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
        self.class_names   = loaded['classes']    
