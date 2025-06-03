import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from models.embedding_model import EmbeddingModel
from preprocess_data import CLASSES


class ViTEmbedder(EmbeddingModel):
    def __init__(self, model_name="google/vit-base-patch16-224-in21k"):
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model_name = model_name
        super().__init__(self.model, model_name=model_name)

    def embed(self, face_tensor: torch.Tensor) -> torch.Tensor:
        # Expect face_tensor to be [3,H,W] or [1,3,H,W] in [-1,1]
        if face_tensor.dim() == 4 and face_tensor.size(0) == 1:
            face_single = face_tensor.squeeze(0)
        elif face_tensor.dim() == 3:
            face_single = face_tensor
        else:
            raise ValueError(f"ViTEmbedder.embed expected [3,H,W] or [1,3,H,W], got {face_tensor.shape}")
        
        # Undo [-1,1] → [0,1]
        face_unnorm = (face_single * 0.5 + 0.5).clamp(0, 1)
        face_np = face_unnorm.permute(1, 2, 0).cpu().numpy()
        face_pil = Image.fromarray((face_np * 255).astype(np.uint8))

        inputs = self.processor(images=face_pil, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            emb = outputs.last_hidden_state[:, 0, :]  # shape: [1, D]

        # Normalize and send to correct device
        emb = F.normalize(emb, dim=1)
        return emb.to(face_tensor.device)

    def cos_forward(self, x):
        class_means = self.class_means
        emb = self.embed(x).squeeze(0).cpu().numpy()
        sims = {c: np.dot(emb, class_means[c]) for c in CLASSES}
        pred = max(sims, key=sims.get)
        cosval = sims[pred]
        return pred, cosval

    def compute_accuracy_with_cos(self, dataset, threshold): 
        correct = 0
        cos = 0
        total = 0
        for image, label in dataset: 
            celebrity = CLASSES[label]
            pred, val = self.cos_forward(image)
            if(celebrity == pred):
                correct += 1
            if(celebrity == pred) and (val >= threshold): 
                cos += 1
            total += 1
        accreg = (correct / total) * 100
        acccos = (cos / total) * 100
        return accreg, acccos
    
    def compute_gradient(self, image, celebrity):
        # Resize to 224x224 for ViT
        image_resized = F.interpolate(image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        
        device = next(self.model.parameters()).device
        
        cm_torch = torch.stack(
            [torch.tensor(self.class_means[c], dtype=torch.float32) for c in self.classes],
            dim=1
        ).to(device)
        
        x = image_resized.clone().detach().to(device).requires_grad_(True)

        # Forward pass through ViT
        outputs = self.model(pixel_values=x)
        emb = outputs.last_hidden_state[:, 0, :]  # CLS token
        embn = F.normalize(emb, p=2, dim=1)
        logits = embn @ cm_torch

        y_true = self.classes.index(celebrity)
        label = torch.tensor([y_true], dtype=torch.long).to(device)

        loss = F.cross_entropy(logits, label)
        loss.backward()

        grad_resized = F.interpolate(x.grad, size=(160, 160), mode='bilinear', align_corners=False)
        return grad_resized.squeeze(0)  # → shape [3,160,160]

