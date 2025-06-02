from models.arcface2 import ArcFace
from preprocess_data import TEST_SET


arc_model = ArcFace()

path = "./checkpoints/arcface.npy"

arc_model.load(path)


acc = arc_model.compute_accuracy(TEST_SET)

print (acc)