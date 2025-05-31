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


def plot_lines(x, ys, title, xlabel, ylabel, labels = None, **plot_kwargs):#plot keyword args
    # Determine if ys is a single sequence of scalars or already a list of sequences
    # If the first element of ys is not iterable, wrap it into a list.
    if hasattr(ys, "__iter__"):
        first_elem = next(iter(ys))
        if not hasattr(first_elem, "__iter__"):
            ys = [ys]
    else:
        ys = [ys]

    plt.figure(figsize=(8,6))
    
    for idx, y in enumerate(ys):
        x_vals = x if x is not None else list(range(len(y)))
        
        if labels and idx < len(labels):
            plt.plot(x_vals, y, label=labels[idx], **plot_kwargs)
        else:
            plt.plot(x_vals, y, **plot_kwargs)

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if labels:
        plt.legend()
    
    plt.grid(True)

    plt.show()


