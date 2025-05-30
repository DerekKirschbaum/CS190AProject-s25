from perturbations import perturb_dataset
from vgg import *
from simplecnn import *

from datasets import train_set, test_set

vgg_path = './models/vgg.npy'
cnn_path = './models/simplecnn.pth'


if __name__ == "__main__":

   vgg_model = VGG()
   cnn_model = SimpleCNN()

   vgg_model.build(train_set = train_set, save_path = vgg_path)
   vgg_model.load(vgg_path)
   cnn_model.load(cnn_path)

   accuracy = vgg_model.compute_accuracy(test_set)

   print("baseline accuracy: ", accuracy)
   
   epsilon = 0.25

   perturbed_test_set = perturb_dataset(cnn_model, test_set, epsilon, attack = 'fgsm')

   perturbed_accuracy = vgg_model.compute_accuracy(perturbed_test_set)

   print("perturbed transfer accuracy: ", perturbed_accuracy)
   
    

   




    
