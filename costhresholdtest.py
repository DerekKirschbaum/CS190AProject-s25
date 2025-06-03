# imports
from models.simplecnn import SimpleCNN
from models.vgg import VGG
from models.casia import Casia
from models.arcface import ArcFace
from models.VITEmbeddings import ViTEmbedder
from models.linear import Linear
from perturbations.perturbations import evaluate_attack_cos

from preprocess_data import TEST_SET, TRAIN_SET

figure_path = './figures/costhresholdtests'
if __name__ == "__main__":
    # Instantiate and load each model
    cnn = SimpleCNN()
    vgg = VGG()
    casia = Casia()
    arcface = ArcFace()
    vit = ViTEmbedder()
    linear = Linear()

    cnn_path = "./checkpoints/simplecnn.npy"
    vgg_path = "./checkpoints/vgg.npy"
    casia_path = "./checkpoints/casia.npy"
    arc_path = "./checkpoints/arcface.npy"
    vit_path = "./checkpoints/vit.npy"

    cnn.load(cnn_path)
    vgg.load(vgg_path)
    casia.load(casia_path)
    arcface.load(arc_path)

    vit.load(vit_path)

    # Prepare the list of target models and their labels
    target_models = [cnn, vgg, casia, arcface, vit]
    model_labels  = ["CNN", "VGG", "Casia", "ArcFace", "VIT"]
    attacks = ["noise", "universal", "pgd"]

    # Define the epsilons to test
    epsilons = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]  # [0.0, 0.05, 0.10, ..., 0.45]

    for attack in attacks:
        evaluate_attack_cos(
            source_model=vit,
            target_models=target_models,
            model_labels=model_labels,
            dataset=TEST_SET,
            epsilons=epsilons,
            attack_method= attack,
            save_path=figure_path
        )