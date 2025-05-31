from facenet_pytorch import  InceptionResnetV1
from Models.embedding_model import EmbeddingModel


class VGG(EmbeddingModel): 
    def __init__(self): 
        super().__init__(model = InceptionResnetV1(pretrained='vggface2').eval(), model_name = 'VGG')

