from facenet_pytorch import InceptionResnetV1
from Models.embedding_model import EmbeddingModel


class Casia(EmbeddingModel): 
    def __init__(self): 
        super().__init__(model = InceptionResnetV1(pretrained='casia-webface').eval(), model_name = 'VGG')
