import torch
import cv2
from insightface.app import FaceAnalysis
from embedding_model import EmbeddingModel

# arcface_model.py

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict
from abc import ABC
from torchvision import transforms

from insightface.app import FaceAnalysis
from models.embedding_model import EmbeddingModel  # your base class


class ArcFaceEmbedding(EmbeddingModel):
    def __init__(self, device: str = "cpu", model_name: str = "ArcFace"):
        """
        device: "cpu" or "cuda:0" (whatever you want to run on).
        model_name: just a string that appears in logs/plots.
        """
        # 1) Create the underlying InsightFace FaceAnalysis app
        #    It will download ArcFace weights (512‐dim) under the hood.
        self.device = device
        self.app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        self.app.prepare(ctx_id=0 if device.startswith("cuda") else -1,
                         det_size=(640, 640))  # adjust det_size if you like

        # 2) We do NOT pass a torch.nn.Module into super().__init__,
        #    because InsightFace’s FaceAnalysis is not a torch.nn.Module.
        #    Instead, we’ll override embed() to call `self.app.get(...)`.
        super().__init__(model=None, model_name=model_name)

        # We still keep `self.mtcnn`, `self.to_tensor`, etc. from base class,
        # but in practice we’ll bypass MTCNN since FaceAnalysis does its own detection.

    def embed(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """
        face_tensor: either [3,H,W] or [B,3,H,W], in torch.float‐tensor form,
        *with values in [-1, +1]* if you used the same Normalize((0.5,…),(0.5,…)) pipeline.
        Returns a torch.Tensor of shape [B, 512], already L2‐normalized.
        """

        # 1) Ensure face_tensor is float in [-1,+1]; convert to CPU numpy BGR
        if face_tensor.dim() == 3:
            # Single image case: add batch‐dim
            x_in = face_tensor.unsqueeze(0)  # [1, 3, H, W]
        elif face_tensor.dim() == 4:
            x_in = face_tensor               # [B, 3, H, W]
        else:
            raise ValueError(f"embed(...) got tensor dim={face_tensor.dim()}, expected 3 or 4.")

        B, C, H, W = x_in.shape

        # 2) Convert from [-1,+1] → [0,255] uint8, and from RGB → BGR
        #    InsightFace expects a NumPy array of shape [H, W, 3], dtype=uint8, in BGR order.
        #    So: (tensor * 0.5 + 0.5) yields [0,1], then *255 → [0..255].
        imgs_bgr = []
        for i in range(B):
            img_norm = x_in[i]                # [3, H, W], floats in [-1,+1]
            img_01 = (img_norm * 0.5) + 0.5    # → [0,1]
            img_255 = (img_01 * 255.0).clamp(0, 255).byte()  # → [0..255] uint8
            # Convert [C,H,W] → [H,W,C], still RGB:
            img_hwc = img_255.permute(1, 2, 0).cpu().numpy()
            # Convert RGB→BGR by simple flip of channels:
            img_bgr = img_hwc[:, :, ::-1].copy()  # H×W×3 uint8 BGR
            imgs_bgr.append(img_bgr)

        # 3) Call FaceAnalysis.get(...) on the batch of B images
        #    It returns a list-of-lists-of-face-objects; we pick the first detection for each image.
        embeddings = []
        for img_bgr in imgs_bgr:
            faces = self.app.get(img_bgr)  # list of Face objects
            if len(faces) == 0:
                # If no face was detected, fallback to a zero‐vector or raise an error.
                # Here we’ll use a zero‐vector of length 512 (and normalize later),
                # but you may want to skip this sample or handle differently.
                emb = torch.zeros(512, dtype=torch.float32)
            else:
                # faces[0].embedding is a NumPy array of length 512
                emb = torch.from_numpy(faces[0].embedding.astype(np.float32))
            embeddings.append(emb)

        # 4) Stack → [B, 512] and L2‐normalize
        emb_tensor = torch.stack(embeddings, dim=0)  # [B, 512]
        emb_tensor = F.normalize(emb_tensor, p=2, dim=1)

        return emb_tensor  # [B, 512]