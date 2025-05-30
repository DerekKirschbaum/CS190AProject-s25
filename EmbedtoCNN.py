from perturbations import perturb_dataset
from vgg import *

from simplecnn import *

from dataset import *

vgg_path = './models/vgg.npy'
cnn_path = './models/simplecnn.pth'


if __name__ == "__main__":

   vgg_model = VGGModel()
   cnn_model = SimpleCNN()
   cnn_model.load(cnn_path)
   vgg_model.load(vgg_path)


   accuracy = cnn_model.compute_accuracy(test_set)

   print("baseline accuracy: ", accuracy)
   
   epsilon = 0.5

   perturbed_test_set = perturb_dataset(vgg_model, test_set, epsilon, attack = 'fgsm')

   perturbed_accuracy = cnn_model.compute_accuracy(perturbed_test_set)

   print("perturbed transfer accuracy: ", perturbed_accuracy)
   
    
