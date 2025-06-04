import os
from models.linear import Linear
from models.simplecnn import SimpleCNN
from models.vgg import VGG
from models.casia import Casia
from models.VITEmbeddings import ViTEmbedder
from preprocess_data import TEST_SET
from perturbations.perturbations import Adversary
from perturbations.utils import save_img

# 1) Define a single output folder and ensure it exists
base_folder = "./pictures/Brad"
os.makedirs(base_folder, exist_ok=True)

# 2) Load all your models as before
linear = Linear()
cnn          = SimpleCNN()
vgg          = VGG()
casia        = Casia()
vit          = ViTEmbedder()

linear_path = "./checkpoints/linear.npy"
cnn_path    = "./checkpoints/simplecnn.npy"
vgg_path    = "./checkpoints/vgg.npy"
casia_path  = "./checkpoints/casia.npy"
vit_path    = "./checkpoints/vit.npy"

linear.load(linear_path)
cnn.load(cnn_path)
vgg.load(vgg_path)
casia.load(casia_path)
vit.load(vit_path)

# 3) Pull out a single test image and label
image, label = TEST_SET[4]

# 4) Create one Adversary (we’ll just demonstrate for vit here)
adv = Adversary(model=casia)
model_name = "casia"


# 5) Create a list of epsilons
epsilons = [round(i * 0.1, 2) for i in range(6)]

# 6) Loop through each attack, build a filename (no extra "/"), and save into base_folder


for eps in epsilons:
    perturbed = adv.fgsm(image, label, eps=eps)
    filename = f"FGSMEps={eps}_{model_name}.png"
    full_path = os.path.join(base_folder, filename)
    save_img(perturbed, path=full_path, title = full_path)
