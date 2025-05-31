#imports
from Models.vgg import VGG
from Models.simplecnn import SimpleCNN
from Models.arcface import ArcFace

from data import TRAIN_SET

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

