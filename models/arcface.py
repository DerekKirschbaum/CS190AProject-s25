import os
import torch
import numpy as np
from torchvision import transforms
from insightface.model_zoo import get_model
from typing import Dict
from preprocess_data import CLASSES
from models.pretrained_embedding import EmbeddingModel  
from insightface.app import FaceAnalysis
import torch.nn.functional as F

#can't generate perturbations with this model as source, because its not a pytorch based implementation, need to find a pytorch based one


class ArcFace(EmbeddingModel):
    def __init__(self, model_name = "arcface"): 
        self.app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))  

        # use a local cache directory for model files
        os.environ['INSIGHTFACE_HOME'] = './.insightface_cache'
        self.model = get_model('buffalo_l', download=True)
        self.model.prepare(ctx_id = -1) 

        super().__init__(self.model, model_name = model_name)
    

    def embed(self, face_tensor: torch.Tensor) -> torch.Tensor:

        if face_tensor.dim() == 4 and face_tensor.size(0) == 1:
            face_single = face_tensor.squeeze(0)   # set tensor to [3, H, W]
        elif face_tensor.dim() == 3:
            face_single = face_tensor              # set tensor to [3, H, W]
        else:
            raise ValueError(
                f"ArcFaceModel.embed expected shape [3,H,W] or [1,3,H,W], "
                f"but got {tuple(face_tensor.shape)}"
            )

        face_unnorm = (face_single * 0.5) + 0.5   # [3, H, W] in [0,1]

        face_np = face_unnorm.permute(1, 2, 0).cpu().numpy()        # [H, W, 3] in [0,1]
        face_np_uint8 = (face_np * 255).astype(np.uint8)            # [H, W, 3], dtype=uint8

        emb_np = self.model.get_feat(face_np_uint8).flatten()
        emb_np = emb_np / np.linalg.norm(emb_np)       

        device = face_tensor.device
        emb_torch = torch.from_numpy(emb_np).unsqueeze(0).to(device).float()  # [1, D] on correct device

        return emb_torch


