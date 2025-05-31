from perturbations import Adversary
from Models.vgg import VGG
from Models.arcface import ArcFace
from preprocess_data import TRAIN_SET, TEST_SET, CLASSES

vgg_path = './models/vgg.npy'
arcface_path = './models/arcface.npy'

if __name__ == "__main__":

    vgg_model = VGG()
    arcface_model = ArcFace()

    vgg_model.load(vgg_path)
    arcface_model.load(arcface_path)

    #print("Baseline accuracy for arcface model:", arcface_model.compute_accuracy(TEST_SET))
   
    epsilon = 0.07

    vgg_adv = Adversary(vgg_model)

    vgg_perturbed_set = vgg_adv.perturb_dataset(TEST_SET, eps = epsilon, attack = 'fgsm')

    perturbed_accuracy = arcface_model.compute_accuracy(vgg_perturbed_set)
    perturbed_accuracy_with_cos = arcface_model.compute_accuracy_with_cos(vgg_perturbed_set, 0.5)

    print("perturbed transfer accuracy: ", perturbed_accuracy)
    print("perturbed transfer accuracy with cos threshold: ", perturbed_accuracy_with_cos)
   