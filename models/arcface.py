import os
import glob
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


import cv2
import numpy as np
from insightface.model_zoo import get_model
from numpy import dot
from numpy.linalg import norm
from typing import Dict
from preprocess_data import CLASSES

class ArcFace():
    def __init__(self): 
        self.model = get_model('buffalo_l', download=True)
        self.model.prepare(ctx_id=-1)
        self.to_tensor = transforms.ToTensor()
        self.class_means: Dict[str, np.ndarray] = {}
        
       
    def build(self, dataset, save_path):
        print("Building ArcFace Embeddings...")
        embeddings_by_class = {name: [] for name in CLASSES}
        for img_tensor, label in dataset:
            name = CLASSES[label]
            emb = self.embed(img_tensor)
            embeddings_by_class[name].append(emb)
        self.class_means = {}
        for name, embs in embeddings_by_class.items():
            mean_emb = np.mean(embs, axis=0)
            self.class_means[name] = mean_emb / np.linalg.norm(mean_emb)
        self.save(save_path)
        print("ArcFace Embeddings Build Complete")

    def forward(self, x): 
        class_means = self.class_means
        emb = self.embed(x)
        emb = emb / np.linalg.norm(emb)
        sims = {c: np.dot(emb, class_means[c]) for c in CLASSES}
        pred = max(sims, key=sims.get)
        cosval = sims[pred]
        # print("Similarity scores:")
        # for celeb in CLASSES:
        #     print(f"  {celeb}: {sims[celeb]:.4f}")
        return pred, cosval
    
    # Predicts by greatest cos sim score
    def compute_accuracy(self, dataset): 
        correct = 0
        total = 0
        for image, label in dataset: 
            celebrity = CLASSES[label]
            pred, _ = self.forward(image)
            if(celebrity == pred): 
                correct += 1
            total += 1
        return (correct / total) * 100
    
    # Counts predictions only if cos sim score is >= threshold
    def compute_accuracy_with_cos(self, dataset, threshold): 
        correct = 0
        total = 0
        for image, label in dataset: 
            celebrity = CLASSES[label]
            pred, val = self.forward(image)
            if(celebrity == pred) and (val >= threshold): 
                correct += 1
            total += 1
        return (correct / total) * 100

    def embed(self, face_tensor: torch.Tensor) -> np.ndarray:
        face_tensor_unnorm = face_tensor * 0.5 + 0.5
        
        # Convert tensor to numpy HWC RGB
        face_np = face_tensor_unnorm.permute(1, 2, 0).cpu().numpy()
        face_np_uint8 = (face_np * 255).astype(np.uint8)
        
        # Get embedding from ArcFace model
        embedding = self.model.get_feat(face_np_uint8).flatten()
        embedding = embedding / np.linalg.norm(embedding)

        return embedding
    
    def save(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, self.class_means)

    def load(self, file_path):
        self.class_means = np.load(file_path, allow_pickle=True).item()