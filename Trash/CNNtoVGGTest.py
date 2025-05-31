from perturbations import Adversary
from vgg import VGG
from simplecnn import SimpleCNN
from data import TEST_SET

vgg_path = './models/vgg.npy'
cnn_path = './models/simplecnn.npy'


if __name__ == "__main__":

   vgg_model = VGG()
   cnn_model = SimpleCNN()

   vgg_model.load(vgg_path)
   cnn_model.load(cnn_path)


   accuracy = vgg_model.compute_accuracy(TEST_SET)

   print("baseline accuracy: ", accuracy)
   
   epsilon = 0.07

   cnn_adv = Adversary(cnn_model)

   cnn_perturbed_set = cnn_adv.perturb_dataset(TEST_SET, eps = epsilon, attack = 'fgsm')

   perturbed_accuracy = vgg_model.compute_accuracy(cnn_perturbed_set)

   print("perturbed transfer accuracy: ", perturbed_accuracy)



      
      

      




      
