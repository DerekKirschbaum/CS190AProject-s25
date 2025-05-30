import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict
from datasets import classes  # Your list of celeb names
from insightface.app import FaceAnalysis


class ArcFace:
    @staticmethod
    def tensor_to_pil(tensor_img: torch.Tensor) -> Image.Image:
        """Convert CHW tensor (C,H,W) to HWC PIL Image (RGB, uint8)"""
        tensor_img = tensor_img.detach().cpu()
        if tensor_img.ndim == 4:
            tensor_img = tensor_img.squeeze(0)
        tensor_img = torch.clamp(tensor_img, 0, 1)  # Ensure [0,1]
        tensor_img = (tensor_img * 255).byte() if tensor_img.max() <= 1 else tensor_img.byte()
        img_np = tensor_img.permute(1, 2, 0).numpy()  # CHW → HWC
        return Image.fromarray(img_np)
    
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
        self.class_means: Dict[str, np.ndarray] = {}

    def forward(self, img: Image.Image):
        emb = self.embed(img)
        if emb is None:
            return "Unknown"
        emb = emb / np.linalg.norm(emb)
        sims = {c: np.dot(emb, self.class_means[c]) for c in classes if c in self.class_means}
        if not sims:
            return "Unknown"
        return max(sims, key=sims.get)

    def build(self, dataset, save_path):
        print("Building ArcFace...")
        embeddings_by_class = {name: [] for name in classes}
        for img_pil, label in dataset:
            name = classes[label]
            emb = self.embed(img_pil)
            if emb is not None:
                embeddings_by_class[name].append(emb)
        for name, embs in embeddings_by_class.items():
            if embs:
                mean_emb = np.mean(embs, axis=0)
                self.class_means[name] = mean_emb / np.linalg.norm(mean_emb)
        self.save(save_path)
        print("ArcFace Build Complete")

    def compute_accuracy(self, dataset):
        correct, total = 0, 0
        for image, label in dataset:
            celebrity = classes[label]
            pred = self.forward(image)
            if celebrity == pred:
                correct += 1
            total += 1
        return (correct / total) * 100 if total > 0 else 0

    def save(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, self.class_means)

    def load(self, file_path):
        self.class_means = np.load(file_path, allow_pickle=True).item()

    def embed(self, img_input) -> np.ndarray:
        if isinstance(img_input, torch.Tensor):
            img_input = self.tensor_to_pil(img_input)

        img_np = np.array(img_input)[:, :, ::-1]  # RGB → BGR
        faces = self.app.get(img_np)
        if not faces:
            return None
        return faces[0].embedding
