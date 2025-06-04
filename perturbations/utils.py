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


def plot_lines(x, ys, title, xlabel, ylabel, save_path, labels = None, **plot_kwargs):#plot keyword args
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

    # Set x-axis ticks to increase by 0.02
    if x is not None and all(isinstance(val, (float, int)) for val in x):
        xtick_min, xtick_max = min(x), max(x)
        plt.xticks(np.arange(xtick_min, xtick_max + 0.02, 0.02))
    
    plt.ylim(-5, 105)

    # Set y-axis ticks and grid lines manually at 0, 20, ..., 100
    plt.yticks(np.arange(0, 101, 20))


    plt.grid(True)

    # Legend outside the plot on the right
    if labels:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(right=0.75)  # Add padding on right side

    # Construct filename with suffix if needed
    base_filename = os.path.join(save_path, title)
    full_filename = base_filename + ".png"
    count = 1
    while os.path.exists(full_filename):
        full_filename = f"{base_filename}_{count}.png"
        count += 1

    plt.savefig(full_filename)
    print(f"[✓] Plot saved to {full_filename}")


