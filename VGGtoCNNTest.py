from perturbations import perturb_dataset
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

   perturbed_test_set = perturb_dataset(vgg_model, test_set, epsilon, attack = 'fgsm')

   perturbed_accuracy = cnn_model.compute_accuracy(perturbed_test_set)

   print("perturbed transfer accuracy: ", perturbed_accuracy)
   
    
