# Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from simplecnn import classes, test_set, load_model, CRITERION, compute_accuracy, compute_gradient

import FGSM_Perturbed_Images_Facenet


FIGURE_PATH = './figures/'


# Displaying an Image

def save_img(img: torch.tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(path, dpi = 100, bbox_inches = "tight", pad_inches = 0)



def perturb_image_fgsm(model, image: torch.Tensor, celebrity: str, epsilon: float): # imagetensor [3,160,160] , celebrity = 'Brad Pitt, Angelina Jolie, ...'
  gradient = compute_gradient(model, image, celebrity)
  perturbed_image = epsilon * torch.sign(gradient) + image.clone()
  perturbed_image = perturbed_image.clamp(-1,1)
  return perturbed_image.detach() #returns a tensor of size [3, 160, 160]

def perturb_image_pgd(model, image: torch.Tensor, celebrity: str, epsilon: float, alpha: float, iters = int): # imagetensor [3,160,160] , celebrity = 'Brad Pitt, Angelina Jolie, ...'
  perturbed_image = image.clone()
  for _ in range(iters): 
    gradient = compute_gradient(model, perturbed_image, celebrity)
    perturbed_image += alpha * gradient.sign()
    perturbed_image = torch.max(torch.min(perturbed_image, image + epsilon), image - epsilon)  # Project to ε-ball
    perturbed_image = torch.clamp(perturbed_image, -1.0, 1.0)  # Keep within valid image range
  return perturbed_image.detach() #returns a tensor of size [3, 160, 160]



def perturb_dataset(model, dataset: TensorDataset, epsilon: float, attack: str, alpha = 0.01, iters = 10): #model, TensorDataset, epsilon (int) 
  perturbed_images = []
  labels = []

  for image, label in dataset:
    celebrity = classes[label]
    if(attack == 'fgsm'):
      perturbed_images.append(perturb_image_fgsm(model, image, celebrity, epsilon))
    elif(attack == 'pgd'): 
      perturbed_images.append(perturb_image_pgd(model, image, celebrity, epsilon, alpha, iters))
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

    epsilon = 0.07

    alpha = 0.1

    iters = 10

    # save_img(epsilon * torch.sign(gradient), path = FIGURE_PATH + 'Epsilon * Sign(Gradient).png',)

    # perturbed_image_fgsm = perturb_image_fgsm(model, image, celebrity, epsilon = epsilon)

    # perturbed_image_pgd = perturb_image_pgd(model, image, celebrity, epsilon, alpha, iters)

    # save_img(perturbed_image_fgsm, path = FIGURE_PATH + 'Perturbed Image FGSM.png')

    # save_img(perturbed_image_pgd, path = FIGURE_PATH + 'Perturbed Image PGD.png')

    # perturbed_dataset_fgsm = perturb_dataset(model, test_set, epsilon = epsilon, attack = 'fgsm')
    # perturbed_dataset_pgd =perturb_dataset(model, test_set, epsilon, 'pgd', alpha, iters )

    # perturbed_fgsm_loader = DataLoader(dataset = perturbed_dataset_fgsm, batch_size = 128)
    # perturbed_pgd_loader = DataLoader(dataset = perturbed_dataset_pgd, batch_size = 128)
    
    # test_loader = DataLoader(dataset = test_set, batch_size = 128)

    # print("perturbed fgsm accuracy", compute_accuracy(model, perturbed_fgsm_loader))

    # print("perturbed pgd accuracy", compute_accuracy(model, perturbed_pgd_loader))

    # print("test_accuracy", compute_accuracy(model, test_loader))







