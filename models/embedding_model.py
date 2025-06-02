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
        print(f"Building {self.model_name}…")
        embeddings_by_class = {name: [] for name in self.classes}

        for img_tensor, label_idx in dataset:
            # img_tensor: shape [3,H,W]
            # We want to run it through embed(), which expects a 4D batch.
            single_batch = img_tensor.unsqueeze(0)       # → [1,3,H,W]
            emb_tensor = self.embed(single_batch)        # → [1, D]
            emb_np = emb_tensor.cpu().numpy()[0]         # extract the single row
            class_name = self.classes[label_idx]
            embeddings_by_class[class_name].append(emb_np)

        # Compute and store NumPy mean‐vector per class:
        for class_name, embs in embeddings_by_class.items():
            mean_emb = np.mean(embs, axis=0)
            self.class_means[class_name] = mean_emb / np.linalg.norm(mean_emb)

        print(f"{self.model_name} Build Complete")

        self.save(save_path)

    def forward(self, x):
        # 1) Get embedding of x → [batch, D]
        emb = self.embed(x)           
        emb = F.normalize(emb, dim=1)             # L2‐norm
        # 2) Stack class_means into a [D, num_classes] tensor
        cm_torch = torch.stack(
            [torch.from_numpy(m).to(x.device).float()
             for m in self.class_means.values()],
            dim=1
        )  # [D, num_classes]
        # 3) Return emb @ cm_torch → [batch, num_classes]
        return emb @ cm_torch
 
    def embed(self, face_tensor: torch.Tensor) -> np.ndarray:
        if face_tensor.dim() == 3:
            # Single example: add batch‐dim
            x_in = face_tensor.unsqueeze(0)   # → [1, 3, H, W]
        elif face_tensor.dim() == 4:
            # Already a batch
            x_in = face_tensor              # → [B, 3, H, W]
        else:
            raise ValueError(
                f"embed(...) expected input of dim 3 or 4, got {face_tensor.dim()}"
            )

        # Pass through the underlying embedding network:
        emb = self.model(x_in)                   # → [batch_size, D]
        emb = F.normalize(emb, p=2, dim=1)       # L2‐normalize each row

        return emb   
    
    def compute_gradient(self, image, celebrity):  # tensor [3,160,160], celebrity = 'scarlett', 'brad',
        # 1) build a stacked tensor of all class‐means: shape (512,5)
        cm_torch = torch.stack([torch.tensor(self.class_means[c], dtype=torch.float32)
                                for c in self.classes], dim=1)
        # 2) prepare input for gradient
        x = image.unsqueeze(0).clone().detach().requires_grad_(True) #(1,3,160,160)
        # 3) forward → embedding → normalize → compute 5 “logits”
        logits = self.forward(x)                              # (1,5)

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

        # We don’t need gradients here:
        with torch.no_grad():
            for image, label in dataset:
                # 1) “image” is [3, H, W]; turn it into a batch [1, 3, H, W]
                x = image.unsqueeze(0)  # → shape [1, 3, H, W]
                #    If your model lives on a GPU, you might need: x = x.to(device)

                # 2) Run it through forward(...) to get [1, num_classes] logits
                logits = self.forward(x)     # → tensor of shape [1, num_classes]

                # 3) Find the predicted index (0 .. num_classes-1)
                pred_idx = int(logits.argmax(dim=1).item())

                # 4) True index:
                true_idx = int(label)  # or label.item() if label is a 0-d torch.Tensor

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
