# Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from simplecnn import classes, test_set, load_model, CRITERION, compute_accuracy


FIGURE_PATH = './figures/'


# Displaying an Image

def save_img(img: torch.tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(path, dpi = 100, bbox_inches = "tight", pad_inches = 0)


def compute_gradient(model, image: torch.Tensor, celebrity: str):  #image: Tensor [3,160,160], celebrity: string, e.g. "Tom Hanks"
    model.eval()
    
    image = image.clone().unsqueeze(0).requires_grad_(True)
    
    idx = classes.index(celebrity)
    label = torch.tensor([idx], dtype=torch.long)

    output = model(image)  
    loss   = CRITERION(output, label)
    loss.backward()
    
    grad = image.grad.detach().squeeze(0)  # [3,160,160]
    grad = grad.clamp(-1,1)
    return grad



def perturb_image_fgsm(model, image: torch.Tensor, celebrity: str, epsilon: float): # imagetensor [3,160,160] , celebrity = 'Brad Pitt, Angelina Jolie, ...'
  gradient = compute_gradient(model, image, celebrity)
  perturbed_image = epsilon * torch.sign(gradient) + image.clone()
  perturbed_image = perturbed_image.clamp(-1,1)
  return perturbed_image.detach() #returns a tensor of size [3, 160, 160]


def perturb_dataset_fgsm(model, dataset, epsilon: float): #model, torch.Dataset, epsilon (int) 
  perturbed_images = []
  labels = []

  for image, label in dataset:
    celebrity = classes[label]
    perturbed_images.append(perturb_image_fgsm(model, image, celebrity, epsilon))
    label = torch.tensor(label)
    labels.append(label)

  perturbed_images = torch.stack(perturbed_images)
  labels = torch.stack(labels)
  return TensorDataset(perturbed_images, labels)


  
if __name__ == "__main__":
    image = test_set[0][0]
    label = test_set[0][1]
    celebrity = classes[label]

    model = load_model()

    save_img(image, path = FIGURE_PATH + 'Original Image.png')
    
    gradient = compute_gradient(model, image, celebrity)

    epsilon = 0.08

    save_img(epsilon * torch.sign(gradient), path = FIGURE_PATH + 'Gradient.png',)

    perturbed_image = perturb_image_fgsm(model, image, celebrity, epsilon = epsilon)

    save_img(perturbed_image, path = FIGURE_PATH + 'Perturbed Image.png')

    perturbed_dataset = perturb_dataset_fgsm(model, test_set, epsilon = epsilon)
    perturbed_loader = DataLoader(dataset = perturbed_dataset, batch_size = 128)
    test_loader = DataLoader(dataset = test_set, batch_size = 128)

    print("perturbed accuracy", compute_accuracy(model, perturbed_loader))

    print("test_accuracy", compute_accuracy(model, test_loader))

    print(perturbed_dataset[0][1])
    print(test_set[0][1])





