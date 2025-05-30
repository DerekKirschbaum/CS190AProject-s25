from perturbations import perturb_dataset
from vgg import load_vgg_means, compute_accuracy_vgg
from simplecnn import load_simple_cnn, test_set


if __name__ == "__main__":

   vgg_model = load_vgg_means()
   cnn_model = load_simple_cnn()
   accuracy = compute_accuracy_vgg(vgg_model, test_set)

   print("baseline accuracy: ", accuracy)
   
   epsilon = 0.25

   perturbed_test_set = perturb_dataset(cnn_model, test_set, epsilon, attack = 'fgsm')

   perturbed_accuracy = compute_accuracy_vgg(vgg_model, perturbed_test_set)

   print("perturbed transfer accuracy: ", perturbed_accuracy)
   
    

   




    
