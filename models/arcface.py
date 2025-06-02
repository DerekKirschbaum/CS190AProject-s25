import os
import torch
import numpy as np
from torchvision import transforms
from insightface.model_zoo import get_model
from typing import Dict
from preprocess_data import CLASSES
from models.embedding_model import EmbeddingModel  
from insightface.app import FaceAnalysis


class ArcFace(EmbeddingModel):
    def __init__(self, model_name = "arcface"): 
        self.app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))  # adjust det_size if you like

        # Use a local cache directory for model files
        os.environ['INSIGHTFACE_HOME'] = './.insightface_cache'
        self.model = get_model('buffalo_l', download=True)
        self.model.prepare(ctx_id = -1)  # Use CPU

        super().__init__(self.model, model_name = model_name)
       

    def embed(self, face_tensor: torch.Tensor) -> torch.Tensor:

        # 1) Handle “batch-of-1” vs “no batch”:
        if face_tensor.dim() == 4 and face_tensor.size(0) == 1:
            face_single = face_tensor.squeeze(0)   # → [3, H, W]
        elif face_tensor.dim() == 3:
            face_single = face_tensor              # → [3, H, W]
        else:
            raise ValueError(
                f"ArcFaceModel.embed expected shape [3,H,W] or [1,3,H,W], "
                f"but got {tuple(face_tensor.shape)}"
            )

        # 2) Undo the [-1,1] → [0,1] scaling:
        #     face_single is in [-1,1], so (x*0.5 + 0.5) puts it into [0,1].
        face_unnorm = (face_single * 0.5) + 0.5   # [3, H, W] in [0,1]

        # 3) Convert to CPU, H×W×3, uint8:
        #    ‣ permute (C, H, W) → (H, W, C), then multiply by 255 and cast.
        face_np = face_unnorm.permute(1, 2, 0).cpu().numpy()        # [H, W, 3] in [0,1]
        face_np_uint8 = (face_np * 255).astype(np.uint8)            # [H, W, 3], dtype=uint8

        # 4) Run the ArcFace model to get a NumPy embedding:
        emb_np = self.model.get_feat(face_np_uint8).flatten()       # shape (D,)
        emb_np = emb_np / np.linalg.norm(emb_np)                    # L2-normalize in NumPy

        # 5) Convert back to a torch.Tensor of shape [1, D] on the same device:
        device = face_tensor.device
        emb_torch = torch.from_numpy(emb_np).unsqueeze(0).to(device).float()  # [1, D] on correct device

        return emb_torch


    def cos_forward(self, x): 
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
