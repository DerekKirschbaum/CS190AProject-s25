from perturbations import Adversary
from vgg import VGG
from simplecnn import SimpleCNN
from datasets import test_set

vgg_path = './models/vgg.npy'
cnn_path = './models/simplecnn.npy'

if __name__ == "__main__":
   vgg_model = VGG()
   cnn_model = SimpleCNN()

   cnn_model.load(cnn_path)
   vgg_model.load(vgg_path)


   accuracy = cnn_model.compute_accuracy(test_set)

   print("baseline accuracy: ", accuracy)
   
   epsilon = 0.07

   vgg_adv = Adversary(cnn_model)

   vgg_perturbed_set = vgg_adv.perturb_dataset(test_set, eps = epsilon, attack = 'fgsm')

   perturbed_accuracy = cnn_model.compute_accuracy(vgg_perturbed_set)

   print("perturbed transfer accuracy: ", perturbed_accuracy)
   
    
