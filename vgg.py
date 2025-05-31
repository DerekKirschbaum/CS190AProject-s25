import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from embedding_model import EmbeddingModel


class VGG(EmbeddingModel): 
    def __init__(self): 
        super().__init__(model = InceptionResnetV1(pretrained='vggface2').eval(), model_name = 'VGG')
        self.mtcnn = MTCNN(image_size=160, margin=0)

    #Helper Methods
    
    def embed(self, face_tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            emb = self.model(face_tensor.unsqueeze(0))
            emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()[0]

