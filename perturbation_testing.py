# imports
from models.simplecnn import SimpleCNN
from models.vgg import VGG
from models.casia import Casia
from perturbations import evaluate_attack

from preprocess_data import TEST_SET

figure_path = './figures/'
if __name__ == "__main__":
    # Instantiate and load each model
    cnn = SimpleCNN()
    vgg = VGG()
    casia = Casia()

    cnn_path = "./checkpoints/simplecnn.npy"
    vgg_path = "./checkpoints/vgg.npy"
    casia_path = "./checkpoints/casia.npy"

    cnn.load(cnn_path)
    vgg.load(vgg_path)
    casia.load(casia_path)

    # Prepare the list of target models and their labels
    target_models = [cnn, vgg, casia]
    model_labels  = ["CNN", "VGG", "Casia"]

    # Define the epsilons to test
    epsilons = [round(i * 0.05, 2) for i in range(10)]  # [0.0, 0.05, 0.10, ..., 0.45]


    evaluate_attack(
        source_model=casia,
        target_models=target_models,
        model_labels=model_labels,
        dataset=TEST_SET,
        epsilons=epsilons,
        attack_method="fgsm",
        save_path=figure_path
    )
    evaluate_attack(
        source_model=casia,
        target_models=target_models,
        model_labels=model_labels,
        dataset=TEST_SET,
        epsilons=epsilons,
        attack_method="noise",
        save_path=figure_path
    )

    evaluate_attack(
        source_model=casia,
        target_models=target_models,
        model_labels=model_labels,
        dataset=TEST_SET,
        epsilons=epsilons,
        attack_method="universal",
        save_path=figure_path
    )

    evaluate_attack(
        source_model=casia,
        target_models=target_models,
        model_labels=model_labels,
        dataset=TEST_SET,
        epsilons=epsilons,
        attack_method="pgd",
        save_path=figure_path
    )




