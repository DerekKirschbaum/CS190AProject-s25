from perturbations import Adversary
from Models.vgg import VGG
from Models.simplecnn import SimpleCNN
from data import TEST_SET

vgg_path = './models/vgg.npy'
cnn_path = './models/simplecnn.npy'

if __name__ == "__main__":
   vgg_model = VGG()
   cnn_model = SimpleCNN()

   cnn_model.load(cnn_path)
   vgg_model.load(vgg_path)


   accuracy = cnn_model.compute_accuracy(TEST_SET)

   print("baseline accuracy: ", accuracy)
   
   epsilon = 0.07

   vgg_adv = Adversary(vgg_model)

   vgg_perturbed_set = vgg_adv.perturb_dataset(TEST_SET, eps = epsilon, attack = 'fgsm')

   perturbed_accuracy = cnn_model.compute_accuracy(vgg_perturbed_set)

   print("perturbed transfer accuracy: ", perturbed_accuracy)
   
    
