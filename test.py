from models.linear import Linear
from preprocess_data import TEST_SET
from perturbations.perturbations import Adversary
from perturbations.utils import save_img


linear_model = Linear()
path = "./checkpoints/linear.npy"
linear_model.load(path)

image = TEST_SET[0][0]
label = TEST_SET[0][1]

adv = Adversary(model = linear_model)


epsilons = [round(i * 0.3, 2) for i in range(10)] 
for eps in epsilons: 
    perturbed = adv.noise(image, label, eps = eps)
    path = "./pictures/NoiseEps = " + str(eps) + ".png" 
    save_img(perturbed, path = path)



