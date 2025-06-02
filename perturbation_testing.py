# imports
from models.simplecnn import SimpleCNN
from models.vgg import VGG
from models.casia import Casia
from models.arcface import ArcFace
from functions.perturbations import evaluate_attack

from preprocess_data import TEST_SET

figure_path = './figures/'
if __name__ == "__main__":
    # Instantiate and load each model
    cnn = SimpleCNN()
    vgg = VGG()
    casia = Casia()
    arcface = ArcFace()

    cnn_path = "./checkpoints/simplecnn.npy"
    vgg_path = "./checkpoints/vgg.npy"
    casia_path = "./checkpoints/casia.npy"
    arc_path = "./checkpoints/arcface.npy"

    cnn.load(cnn_path)
    vgg.load(vgg_path)
    casia.load(casia_path)
    arcface.load(arc_path)

    # Prepare the list of target models and their labels
    target_models = [cnn, vgg, casia, arcface]
    source_models = [cnn, vgg, casia]
    model_labels  = ["CNN", "InceptionResnetV1(VGG)", "InceptionResnetV1(Casia)", "ArcFace"]
    attacks = ["fgsm", "noise", "universal", "pgd"]

    # Define the epsilons to test
    epsilons = [round(i * 0.02, 2) for i in range(10)]  # [0.0, 0.05, 0.10, ..., 0.45]

    for attack in attacks: 
        for model in source_models: 
            evaluate_attack(
                source_model= model,
                target_models=target_models,
                model_labels=model_labels,
                dataset=TEST_SET,
                epsilons=epsilons,
                attack_method= attack,
                save_path=figure_path
            )
    



