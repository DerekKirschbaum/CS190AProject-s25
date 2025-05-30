from perturbations import perturb_dataset
from vgg import *
from simplecnn import *


if __name__ == "__main__":

   vgg_model = VGGModel()
   cnn_model = SimpleCNN()
   vgg_model.load()
   cnn_model.load()

   accuracy = vgg_model.compute_accuracy(test_set)

   print("baseline accuracy: ", accuracy)
   
   epsilon = 0.25

   perturbed_test_set = perturb_dataset(cnn_model, test_set, epsilon, attack = 'fgsm')

   perturbed_accuracy = vgg_model.compute_accuracy(perturbed_test_set)

   print("perturbed transfer accuracy: ", perturbed_accuracy)
   
    

   




    
