# imports
from models.simplecnn import SimpleCNN
from models.vgg import VGG
from models.casia import Casia
from models.arcface import ArcFace
from models.linear import Linear
from models.VITEmbeddings import ViTEmbedder

from perturbations.perturbations import evaluate_attack

from preprocess_data import TEST_SET

figure_path = './figures/'
if __name__ == "__main__":
    # Instantiate and load each model
    cnn = SimpleCNN()
    vgg = VGG()
    casia = Casia()
    arcface = ArcFace()
    linear = Linear()
    vit = ViTEmbedder()

    cnn_path = "./checkpoints/simplecnn.npy"
    vgg_path = "./checkpoints/vgg.npy"
    casia_path = "./checkpoints/casia.npy"
    arc_path = "./checkpoints/arcface.npy"
    linear_path = "./checkpoints/linear.npy"
    vit_path = "./checkpoints/vit.npy"

    cnn.load(cnn_path)
    vgg.load(vgg_path)
    casia.load(casia_path)
    arcface.load(arc_path)
    linear.load(linear_path)
    vit.load(vit_path)

    # Prepare the list of target models and their labels
    target_models = [linear, cnn, vgg, casia, arcface, vit]
    source_models = [linear, cnn, vgg, casia]
    model_labels  = ["Linear", "CNN", "InceptionResnetV1(VGG)", "InceptionResnetV1(Casia)", "ArcFace", "Vit"]
    attacks = ["fgsm", "noise"]

    # Define the epsilons to test
    epsilons = [round(i * 0.1, 2) for i in range(2)]  


    for attack in attacks: 
        for source in source_models: 
            evaluate_attack(
                    source_model= source,
                    target_models=target_models,
                    model_labels=model_labels,
                    dataset=TEST_SET,
                    epsilons=epsilons,
                    attack_method= attack,
                    save_path=figure_path
                    )
            



