from models.vgg import VGG
from preprocess_data import TEST_SET
from perturbations import Adversary


path = "./checkpoints/vgg.npy"


vgg = VGG()
vgg.load(path)

acc = vgg.compute_accuracy(TEST_SET)

print(acc)

adv = Adversary(model = vgg)

perturbed = adv.perturb_dataset(TEST_SET, eps = 0.5, attack = 'universal')

accuracy = vgg.compute_accuracy(perturbed)
print(accuracy)

