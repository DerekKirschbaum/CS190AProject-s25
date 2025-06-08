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

base_folder = "./pictures/Brad"
os.makedirs(base_folder, exist_ok=True)

linear = Linear()
cnn    = SimpleCNN()
vgg    = VGG()
casia  = Casia()
vit    = ViTEmbedder()
tiny   = TinyCNN()

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

# pull out a single test image and label
image, label = TEST_SET[4]
celeb = CLASSES[label]

source_models = [cnn, linear, casia, vgg, vit, tiny]
model_labels  = ["SimpleCNN", "Linear", "ResNet_v1(Casia)", "ResNet_v1(VGG)", "VIT", "TinyCNN"]

epsilons = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
for i in range (len(source_models)): 
    adv = Adversary(model=source_models[i])
    for epsilon in epsilons: 
        perturbed = adv.fgsm(image, label, eps = epsilon)
        folder = base_folder + "/" + model_labels[i]
        os.makedirs(folder, exist_ok=True)
        path = folder + "/Epsilon=" + str(epsilon) + ".png"

        title = "FGSM, Source: " + model_labels[i] + "Epsilon= " + str(epsilon)
        save_img(perturbed, path = path, title = title)




