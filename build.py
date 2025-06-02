#imports
from models.vgg import VGG
from models.simplecnn import SimpleCNN
from models.casia import Casia
from models.arcface import ArcFace
from preprocess_data import TRAIN_SET

if __name__ == "__main__":
    arc_model = ArcFace()
    vgg_model = VGG()
    cnn_model = SimpleCNN()
    casia_model = Casia()

    vgg_path = "./checkpoints/vgg.npy" #path for model to be saved at
    cnn_path = "./checkpoints/simplecnn.npy"
    casia_path = "./checkpoints/casia.npy"
    arc_path = "./checkpoints/arcface.npy"

    # vgg_model.build(dataset = TRAIN_SET, save_path = vgg_path)
    # cnn_model.build(dataset = TRAIN_SET, save_path = cnn_path)
    # casia_model.build(dataset = TRAIN_SET, save_path = casia_path)
    arc_model.build(dataset = TRAIN_SET, save_path = arc_path)

