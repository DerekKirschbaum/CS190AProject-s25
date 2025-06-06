# imports
from models.simplecnn import SimpleCNN
from models.vgg import VGG
from models.casia import Casia
from models.arcface import ArcFace
from models.linear import Linear
from models.tinycnn import TinyCNN
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
    tinycnn = TinyCNN()

    cnn_path = "./checkpoints/simplecnn.npy"
    vgg_path = "./checkpoints/vgg.npy"
    casia_path = "./checkpoints/casia.npy"
    arc_path = "./checkpoints/arcface.npy"
    linear_path = "./checkpoints/linear.npy"
    vit_path = "./checkpoints/vit.npy"
    tiny_path = "./checkpoints/tiny.npy"
    

    cnn.load(cnn_path)
    vgg.load(vgg_path)
    casia.load(casia_path)
    arcface.load(arc_path)
    linear.load(linear_path)
    vit.load(vit_path)
    tinycnn.load(tiny_path)

    # Prepare the list of target models and their labels
    target_models = [cnn, linear, casia, vgg, arcface, vit]
    source_models = [cnn, linear, casia, vgg, vit]
    model_labels  = ["SimpleCNN", "Linear", "ResNet_v1(Casia)", "ResNet_v1(VGG)", "ArcFace", "VIT"]
    attacks = ["noise"]

    # Define the epsilons to test
    epsilons = [0.12]
    
    for attack in attacks:
        for source_model in source_models:
            evaluate_attack(
                source_model = source_model,
                target_models=target_models,
                model_labels=model_labels,
                dataset=TEST_SET,
                epsilons=epsilons,
                attack_method= attack,
                save_path=figure_path
            )
  
                



