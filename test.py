from preprocess_data import TEST_SET
from models.linear import Linear
from models.vgg import VGG
import torch.nn as nn
from perturbations.perturbations import Adversary

vgg_path = "./checkpoints/vgg.npy"
linear_path = "./checkpoints/linear.npy"

vgg = VGG()
linear = Linear()

vgg.load(vgg_path)
linear.load(linear_path)
for i in range(10): 
    print("IMAGE: ", i)
    image = TEST_SET[i][0]
    label = TEST_SET[i][1]
    batch = image.unsqueeze(dim = 0)


    softmax = nn.Softmax(dim=1)

    print("Regular")

    pred = vgg.forward(batch)
    print("vgg pred: ", pred)

    pred = linear.forward(batch)
    print("linear pred: ", softmax(pred))

    adv = Adversary(model = linear)

    perturbed = adv.fgsm(image, label, eps = 3)
    batch = perturbed.unsqueeze(dim = 0)

    print("Perturbed")

    pred = vgg.forward(batch)
    print("vgg pred: ", pred)

    pred = linear.forward(batch)
    print("linear pred: ", softmax(pred))






