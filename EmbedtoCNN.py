from perturbations import perturb_dataset
from vgg import load_vgg_means
from simplecnn import load_simple_cnn, test_set, classes, compute_accuracy_cnn


if __name__ == "__main__":

   vgg_model = load_vgg_means()
   cnn_model = load_simple_cnn()
   accuracy = compute_accuracy_cnn(cnn_model, test_set)

   print("baseline accuracy: ", accuracy)
   
   epsilon = 0.5

   perturbed_test_set = perturb_dataset(vgg_model, test_set, epsilon, attack = 'fgsm', is_embed = True)

   perturbed_accuracy = compute_accuracy_cnn(cnn_model, perturbed_test_set)

   print("perturbed transfer accuracy: ", perturbed_accuracy)
   
    
