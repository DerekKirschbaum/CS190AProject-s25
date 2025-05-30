import os
import numpy as np
import matplotlib.pyplot as plt


FIGURE_PATH = './Figures/'

def save_img(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(path, dpi = 100, bbox_inches = "tight", pad_inches = 0)
