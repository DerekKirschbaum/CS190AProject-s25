import os
from models.linear import Linear
from models.simplecnn import SimpleCNN
from models.vgg import VGG
from models.casia import Casia
from models.VITEmbeddings import ViTEmbedder
from models.tinycnn import TinyCNN
from preprocess_data import TEST_SET, CLASSES
from perturbations.perturbations import Adversary
from perturbations.utils import save_img
import torch

# 1) Define a single output folder and ensure it exists
base_folder = "./pictures/Gradients"
os.makedirs(base_folder, exist_ok=True)

# 2) Load all your models as before
linear = Linear()
cnn          = SimpleCNN()
vgg          = VGG()
casia        = Casia()
vit          = ViTEmbedder()
tiny = TinyCNN()

linear_path = "./checkpoints/linear.npy"
cnn_path    = "./checkpoints/simplecnn.npy"
vgg_path    = "./checkpoints/vgg.npy"
casia_path  = "./checkpoints/casia.npy"
vit_path    = "./checkpoints/vit.npy"
tiny_path = "./checkpoints/tiny.npy"

linear.load(linear_path)
cnn.load(cnn_path)
vgg.load(vgg_path)
casia.load(casia_path)
vit.load(vit_path)
tiny.load(tiny_path)

# 3) Pull out a single test image and label
image, label = TEST_SET[0]
celeb = CLASSES[label]

adv = Adversary(model=cnn)
model_name = "TinyCNN"

gradient = adv.noise(image, label, eps = 1)
save_img(torch.sign(gradient) - image, path = "./pictures/Gradients/Noise.png", title = "Noise")

rand = torch.rand_like(image) * 2.0 - 1.0
save_img(rand, path = "./pictures/Gradients/Noise.png", title = "Noise")




