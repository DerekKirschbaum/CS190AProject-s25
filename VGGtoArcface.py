from perturbations import Adversary
from vgg import VGG
from simplecnn import SimpleCNN
from datasets import train_set, test_set
from newarcface import ArcFace

vgg_path = './models/vgg.npy'
arcface_path = './models/newarcface.npy'

if __name__ == "__main__":
    vgg_model = VGG()
    arcface_model = ArcFace()

    arcface_model.load(arcface_path)
    vgg_model.load(vgg_path)


    accuracy = arcface_model.compute_accuracy(test_set)

    print("baseline accuracy: ", accuracy)

    epsilon = 0.07

    vgg_adv = Adversary(vgg_model)

    vgg_perturbed_set = vgg_adv.perturb_dataset(test_set, eps = epsilon, attack = 'fgsm')

    perturbed_accuracy = arcface_model.compute_accuracy(vgg_perturbed_set)

    print("perturbed transfer accuracy: ", perturbed_accuracy)