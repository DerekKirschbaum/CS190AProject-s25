from facenet_pytorch import  InceptionResnetV1
from models.pretrained_embedding import EmbeddingModel


class VGG(EmbeddingModel): 
    def __init__(self, model_name = 'VGG'): 
        super().__init__(model = InceptionResnetV1(pretrained='vggface2').eval(), model_name = model_name)

