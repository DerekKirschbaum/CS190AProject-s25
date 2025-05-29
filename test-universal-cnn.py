import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


import vgg
import simplecnn
import perturbations

from perturbations import * 
from vgg import *
from simplecnn import *

if __name__ == "__main__":
    image = test_set[0][0]
    label = test_set[0][1]
    celebrity = classes[label]

    cnn_model = load_simple_cnn()

    save_img(image, path = FIGURE_PATH + 'Original Image.png')
    
    gradient = compute_gradient(model, image, celebrity)

    epsilon = 0.14

    alpha = 0.1

    iters = 10

    print("Computing universal perturbation...")
    v = universal_perturbation(cnn_model, test_set, celebrity, epsilon=epsilon, alpha=alpha, iters=iters)

    # Apply universal perturbation to one image
    universal_image = image + v
    universal_image = torch.max(torch.min(universal_image, image + epsilon), image - epsilon)
    universal_image = torch.clamp(universal_image, -1.0, 1.0)

    save_img(universal_image, path=FIGURE_PATH + 'Universal Perturbed Image.png')

    # Evaluate model accuracy on perturbed dataset
    print("Creating perturbed dataset with universal perturbation...")
    perturbed_universal_dataset = perturb_dataset(cnn_model, test_set, epsilon, attack='universal', alpha=alpha, iters=iters)
    perturbed_universal_loader = DataLoader(perturbed_universal_dataset, batch_size=128)

    print("Computing accuracy on universally perturbed dataset...")
    acc = compute_accuracy_cnn(cnn_model, perturbed_universal_loader)
    print(f"Accuracy on universally perturbed test set: {acc:.4f}")

# === Plot original image, perturbation, and perturbed image ===
def show_comparison(original, perturbation, perturbed):
    """
    Display original image, perturbation, and perturbed image side-by-side.
    """
    # Convert tensors to displayable format
    def to_numpy(img_tensor):
        img = img_tensor.clone().detach()
        img = img / 2 + 0.5  # Unnormalize if necessary
        return img.numpy()

    orig_np = to_numpy(original)
    pert_np = to_numpy(perturbation)
    adv_np = to_numpy(perturbed)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(np.transpose(orig_np, (1, 2, 0)))
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(np.transpose(pert_np, (1, 2, 0)))
    axs[1].set_title("Perturbation (v)")
    axs[1].axis('off')

    axs[2].imshow(np.transpose(adv_np, (1, 2, 0)))
    axs[2].set_title("Perturbed Image")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

# Plot the comparison
show_comparison(image, v, universal_image)