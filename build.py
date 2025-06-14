from models.vgg import VGG
from models.simplecnn import SimpleCNN
from models.linear import Linear
from models.casia import Casia
from models.arcface import ArcFace
from models.VITEmbeddings import ViTEmbedder
from models.tinycnn import TinyCNN
from preprocess_data import TRAIN_SET

if __name__ == "__main__":
    arc_model = ArcFace()
    vgg_model = VGG()
    cnn_model = SimpleCNN()
    casia_model = Casia()
    linear_model = Linear()
    vit_model = ViTEmbedder()
    tiny_model = TinyCNN()
    

    vgg_path = "./checkpoints/vgg.npy" 
    cnn_path = "./checkpoints/simplecnn.npy"
    casia_path = "./checkpoints/casia.npy"
    arc_path = "./checkpoints/arcface.npy"
    linear_path = "./checkpoints/linear.npy"
    vit_path = "./checkpoints/linear.npy"
    tiny_path = "./checkpoints/tiny.npy"


    vgg_model.build(dataset = TRAIN_SET, save_path = vgg_path)
    cnn_model.build(dataset = TRAIN_SET, save_path = cnn_path)
    casia_model.build(dataset = TRAIN_SET, save_path = casia_path)
    arc_model.build(dataset = TRAIN_SET, save_path = arc_path)
    linear_model.build(TRAIN_SET, save_path = linear_path)
    vit_model.build(TRAIN_SET, save_path = linear_path)
    tiny_model.build(TRAIN_SET, save_path = tiny_path, epochs = 15, is_verbose = True)
    

