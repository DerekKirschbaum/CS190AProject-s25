# Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import simplecnn

from simplecnn import classes, test_set
FIGURE_PATH = './figures/'


# Displaying an Image

def save_img(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(path, dpi = 100, bbox_inches = "tight", pad_inches = 0)


def compute_gradient(model, image, celebrity): # image [3,160,160] , celebrity = 'Brad Pitt, Angelina Jolie, ...'
  image.requires_grad_()
  label = classes.index(celebrity)

  outputs = model.forward(image)
  loss = simplecnn.CRITERION(outputs, label)
  loss.backward()
  grad = image.grad  
  return grad   #returns a tensor of size [3, 160, 160]



if __name__ == "__main__":

    image = test_set[0][0]
    print(image.shape)
    label = test_set[0][1]
    celebrity = classes[label]
    save_img(image, path = FIGURE_PATH + 'Original Image.png')

