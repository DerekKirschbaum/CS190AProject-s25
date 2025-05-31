import numpy as np
import matplotlib.pyplot as plt

from perturbations import Adversary 
from vgg import VGG
from simplecnn import SimpleCNN
from data import TEST_SET, CLASSES
from utils import save_img, FIGURE_PATH


if __name__ == "__main__":
    image = TEST_SET[0][0]
    label = TEST_SET[0][1]
    celebrity = CLASSES[label]

    model = SimpleCNN()
    model.load(file_path = './models/simplecnn.npy')

    save_img(image, path = FIGURE_PATH + 'Original Image.png')
    
    gradient = model.compute_gradient(image, celebrity)

    epsilon = 0.14

    alpha = 0.1

    iters = 10

    print("Computing universal perturbation...")
    cnn_adv = Adversary(model)

    perturbed_universal_dataset = cnn_adv.perturb_dataset(TEST_SET, eps = epsilon, attack = 'universal')

    v = cnn_adv.make_universal(TEST_SET, epsilon)

    # Apply universal perturbation to one image
    universal_image = cnn_adv.universal(image, v, epsilon)

    save_img(universal_image, path=FIGURE_PATH + 'Universal Perturbed Image.png')

    # Evaluate model accuracy on perturbed dataset
    print("Creating perturbed dataset with universal perturbation...")

    print("Computing accuracy on universally perturbed dataset...")
    acc = model.compute_accuracy( perturbed_universal_dataset)
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