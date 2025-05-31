#imports
from simplecnn import SimpleCNN
from perturbations import Adversary
from utils import plot_lines

from data import TEST_SET

if __name__ == "__main__":
    cnn = SimpleCNN()
    path = "./models/simplecnn.npy"
    cnn.load(path)

    adv = Adversary(cnn)

    epsilons = [round(i * 0.05, 2) for i in range(11)]
    accuracy = []

    for eps in epsilons: 
        perturbed_set = adv.perturb_dataset(TEST_SET, eps, 'fgsm')
        accuracy.append(cnn.compute_accuracy(perturbed_set))
    

    
    plot_lines(epsilons, accuracy, "Accuracy vs Epsilon", "Epsilon", "Accuracy")
