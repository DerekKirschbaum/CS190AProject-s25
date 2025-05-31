#imports
from models.simplecnn import SimpleCNN
from models.vgg import VGG
from models.casia import Casia
from perturbations import Adversary
from utils import plot_lines

from preprocess_data import TEST_SET

figure_path = './figures/'

if __name__ == "__main__":
    cnn = SimpleCNN()
    vgg = VGG()
    casia = Casia()


    cnn_path = "./checkpoints/simplecnn.npy"
    vgg_path = "./checkpoints/vgg.npy"
    casia_path = "./checkpoints/casia.npy"

    cnn.load(cnn_path)
    vgg.load(vgg_path)
    casia.load(casia_path)


    adv = Adversary(cnn)

    epsilons = [round(i * 0.05, 2) for i in range(10)]
    cnn_accuracy = []
    vgg_accuracy = []
    casia_accuracy = []

    for eps in epsilons: 
        perturbed_set = adv.perturb_dataset(TEST_SET, eps, 'noise')
        cnn_accuracy.append(cnn.compute_accuracy(perturbed_set))
        vgg_accuracy.append(vgg.compute_accuracy(perturbed_set))
        casia_accuracy.append(casia.compute_accuracy(perturbed_set))
    

    
    plot_lines(epsilons, [cnn_accuracy,vgg_accuracy, casia_accuracy], "FGSM attack using Noise Perturbation", "Epsilon", "Accuracy",labels = ["CNN","VGG","Casia"], save_path = figure_path)
