#imports

from vgg import VGG
from simplecnn import SimpleCNN

from datasets import train_set

if __name__ == "__main__":
    vgg_model = VGG()
    cnn_model = SimpleCNN()

    vgg_path = "./models/vgg"
    cnn_path = "./models/simplecnn"


    vgg_model.build(dataset = train_set, save_path = vgg_path)
    cnn_model.build(dataset = train_set, save_path = cnn_path)

