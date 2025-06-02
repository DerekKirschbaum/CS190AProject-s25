# imports
from models.simplecnn import SimpleCNN
from models.vgg import VGG
from models.casia import Casia
from models.arcface import ArcFace
from perturbations import evaluate_attack

from preprocess_data import TEST_SET, CLASSES

figure_path = './figures/'
if __name__ == "__main__":
   arc = ArcFace()
   vgg = VGG()
   path = "./checkpoints/arcface.npy"
   vgg_path = "./checkpoints/vgg.npy"
   arc.load(path)
   vgg.load(vgg_path)

   image = TEST_SET[0][0]
   celeb = CLASSES[TEST_SET[0][1]]

   grad = vgg.compute_gradient(image, celeb)
   print(grad)
   grad = arc.compute_gradient(image, celeb)

   print(grad)