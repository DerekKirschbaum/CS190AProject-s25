from perturbations import perturb_dataset
from vgg import load_vgg_means

from simplecnn import *
from dataset import *


if __name__ == "__main__":

   vgg_model = load_vgg_means()
   cnn_model = SimpleCNN()
   cnn_model.load()

   accuracy = model.compute_accuracy(test_set)

   print("baseline accuracy: ", accuracy)
   
   epsilon = 0.5

   perturbed_test_set = perturb_dataset(vgg_model, test_set, epsilon, attack = 'fgsm', is_embed = True)

   perturbed_accuracy = model.compute_accuracy(perturbed_test_set)

   print("perturbed transfer accuracy: ", perturbed_accuracy)
   
    
