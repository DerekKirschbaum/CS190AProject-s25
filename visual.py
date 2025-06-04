from models.linear import Linear
from models.simplecnn import SimpleCNN
from models.vgg import VGG
from models.casia import Casia
from models.VITEmbeddings import ViTEmbedder
from preprocess_data import TEST_SET
from perturbations.perturbations import Adversary
from perturbations.utils import save_img


linear_model = Linear()
cnn = SimpleCNN()
vgg = VGG()
casia = Casia()
vit = ViTEmbedder()
path = "./checkpoints/linear.npy"
cnn_path = "./checkpoints/simplecnn.npy"
vgg_path = "./checkpoints/vgg.npy"
casia_path = "./checkpoints/casia.npy"
vit_path = "./checkpoints/vit.npy"


linear_model.load(path)
cnn.load(cnn_path)
vgg.load(vgg_path)
casia.load(casia_path)
vit.load(vit_path)

image = TEST_SET[0][0]
label = TEST_SET[0][1]

adv = Adversary(model = vit)

epsilons = [round(i * 0.04, 2) for i in range(6)] 
for eps in epsilons: 
    perturbed = adv.noise(image, label, eps = eps)
    path = "./pictures/NoiseEps = " + str(eps) + ".png" 
    save_img(perturbed, path = path)

for eps in epsilons: 
    perturbed = adv.fgsm(image, label, eps = eps)
    path = "./pictures/FGSMEps = " + str(eps) + ".png" 
    save_img(perturbed, path = path)

for eps in epsilons: 
    perturbed = adv.pgd(image, label, eps = eps)
    path = "./pictures/PGDEps = " + str(eps) + ".png" 
    save_img(perturbed, path = path)






