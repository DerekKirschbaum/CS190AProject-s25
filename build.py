#imports

from vgg import VGG
from simplecnn import SimpleCNN
from arcface import ArcFace

from data import TRAIN_SET

if __name__ == "__main__":
    vgg_model = VGG()
    cnn_model = SimpleCNN()
    arcface_model = ArcFace()

    vgg_path = "./models/vgg.npy" #path for model to be saved at
    cnn_path = "./models/simplecnn.npy"
    arcface_path = "./models/arcface.npy"


    vgg_model.build(dataset = TRAIN_SET, save_path = vgg_path)
    cnn_model.build(dataset = TRAIN_SET, save_path = cnn_path)
    arcface_model.build(dataset = TRAIN_SET, save_path = arcface_path)

