# Imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from simplecnn import classes, test_set, load_simple_cnn, CRITERION, compute_accuracy_cnn, compute_gradient

import vgg


FIGURE_PATH = './figures/'


# Displaying an Image


def save_img(img: torch.tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(path, dpi = 100, bbox_inches = "tight", pad_inches = 0)



def perturb_image_fgsm(model, image: torch.Tensor, celebrity: str, epsilon: float, is_embed: bool = False): # imagetensor [3,160,160] , celebrity = 'Brad Pitt, Angelina Jolie, ...'
  if(is_embed == True): 
    gradient =  vgg.compute_gradient(model, image, celebrity)
  elif (is_embed == False):
    gradient = compute_gradient(model, image, celebrity)
  perturbed_image = epsilon * torch.sign(gradient) + image.clone()
  perturbed_image = perturbed_image.clamp(-1,1)
  return perturbed_image.detach() #returns a tensor of size [3, 160, 160]


def perturb_image_pgd(model, image: torch.Tensor, celebrity: str, epsilon: float, alpha: float, iters = int, is_embed = False): # imagetensor [3,160,160] , celebrity = 'Brad Pitt, Angelina Jolie, ...'
  perturbed_image = image.clone()
  for _ in range(iters): 
    if(is_embed == True): 
      gradient =  vgg.compute_gradient(model, image, celebrity)
    elif (is_embed == False):
      gradient = compute_gradient(model, perturbed_image, celebrity)
    perturbed_image += alpha * gradient.sign()
    perturbed_image = torch.max(torch.min(perturbed_image, image + epsilon), image - epsilon)  # Project to ε-ball
    perturbed_image = torch.clamp(perturbed_image, -1.0, 1.0)  # Keep within valid image range
  return perturbed_image.detach() #returns a tensor of size [3, 160, 160]

def universal_perturbation(model, dataset: TensorDataset, celebrity: str, epsilon: float, alpha: float, iters = int): 
  v = torch.zeros_like(dataset[0][0])
  for _ in range(iters):
        for image, label in dataset:
          image = image.to(torch.float32)
          perturbed_image = image + v
          perturbed_image = torch.max(torch.min(perturbed_image, image + epsilon), image - epsilon)  # Project to ε-ball
          perturbed_image = torch.clamp(perturbed_image, -1.0, 1.0)

          #Run image through the model
          celebrity = classes[label]
          pred_logits = model(perturbed_image.unsqueeze(0))  # batch of 1
          pred_label = torch.argmax(pred_logits, dim=1)

          if pred_label.item() == label: #condition to update v
                gradient = compute_gradient(model, perturbed_image, celebrity)
                v = v + (alpha * gradient.sign())
                v = torch.max(torch.min(v, epsilon * torch.ones_like(v)), -epsilon * torch.ones_like(v))

  return v.detach()  #tensor [3,160,160]


def perturb_dataset(model, dataset: TensorDataset, epsilon: float, attack: str, alpha = 0.01, iters = 10, is_embed = False): #model, TensorDataset, epsilon (int) 
  perturbed_images = []
  labels = []

  for image, label in dataset:
    celebrity = classes[label]
    if(attack == 'fgsm'):
      perturbed_images.append(perturb_image_fgsm(model, image, celebrity, epsilon, is_embed))
    elif(attack == 'pgd'): 
      perturbed_images.append(perturb_image_pgd(model, image, celebrity, epsilon, alpha, iters, is_embed))
    label = torch.tensor(label)
    labels.append(label)

  if attack == 'universal':
    # Compute the universal perturbation v (shared across all images)
    v = universal_perturbation(model, dataset, celebrity, epsilon=epsilon, alpha=alpha, iters=iters)
    
    # Clear previously appended lists
    perturbed_images = []
    labels = []

    for image, label in dataset:
      perturbed = image + v
      perturbed = torch.max(torch.min(perturbed, image + epsilon), image - epsilon)  # Project to ε-ball
      perturbed = torch.clamp(perturbed, -1.0, 1.0)
      perturbed_images.append(perturbed)
      labels.append(torch.tensor(label))

  perturbed_images = torch.stack(perturbed_images)
  labels = torch.stack(labels)
  return TensorDataset(perturbed_images, labels)








