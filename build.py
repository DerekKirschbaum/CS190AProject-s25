#imports
from models.vgg import VGG
from models.simplecnn import SimpleCNN
from models.arcface import ArcFace

from preprocess_data import TRAIN_SET

if __name__ == "__main__":
    vgg_model = VGG()
    cnn_model = SimpleCNN()
    arcface_model = ArcFace()

    vgg_path = "./checkpoints/vgg.npy" #path for model to be saved at
    cnn_path = "./checkpoints/simplecnn.npy"
    arcface_path = "./checkpoints/arcface.npy"


    vgg_model.build(dataset = TRAIN_SET, save_path = vgg_path)
    cnn_model.build(dataset = TRAIN_SET, save_path = cnn_path)
    arcface_model.build(dataset = TRAIN_SET, save_path = arcface_path)

