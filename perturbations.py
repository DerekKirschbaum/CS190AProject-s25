# Imports
import torch
from torch.utils.data import TensorDataset
from simplecnn import classes





def epsilon_clamp(image, perturbed_image, epsilon): 
    perturbed_image = torch.max(torch.min(perturbed_image, image + epsilon), image - epsilon)  # Project to ε-ball
    perturbed_image = torch.clamp(perturbed_image, -1.0, 1.0)  # Keep within valid image range
    return perturbed_image


def perturb_image_fgsm(model, image: torch.Tensor, celebrity: str, epsilon: float): # imagetensor [3,160,160] , celebrity = 'Brad Pitt, Angelina Jolie, ...'
  gradient = model.compute_gradient(image, celebrity)
  perturbed_image = epsilon * torch.sign(gradient) + image.clone()
  perturbed_image = epsilon_clamp(image, perturbed_image, epsilon)
  return perturbed_image.detach() #returns a tensor of size [3, 160, 160]


def perturb_image_pgd(model, image: torch.Tensor, celebrity: str, epsilon: float, alpha: float, iters = int): # imagetensor [3,160,160] , celebrity = 'Brad Pitt, Angelina Jolie, ...'
  perturbed_image = image.clone()
  for _ in range(iters): 
    gradient =  model.compute_gradient(image, celebrity)
    perturbed_image += alpha * gradient.sign()
    perturbed_image = epsilon_clamp(image, perturbed_image, epsilon)
  return perturbed_image.detach() #returns a tensor of size [3, 160, 160]

def perturb_image_universal(image, v, epsilon): 
  perturbed_image = image + v
  perturbed_image = epsilon_clamp(image, perturbed_image, epsilon)
  return perturbed_image


def generate_universal_perturbation(model, dataset: TensorDataset, epsilon: float, alpha: float, iters = int): 
  v = torch.zeros_like(dataset[0][0])
  for _ in range(iters):
        for image, label in dataset:
          image = image.to(torch.float32)
          perturbed_image = image + v
          perturbed_image = epsilon_clamp(image, perturbed_image, epsilon)

          #Run image through the model
          celebrity = classes[label]
          pred_logits = model(perturbed_image.unsqueeze(0))  # batch of 1
          pred_label = torch.argmax(pred_logits, dim=1)

          if pred_label.item() == label: #condition to update v
              gradient = model.compute_gradient(perturbed_image, celebrity) 
              v = v + (alpha * gradient.sign())
              v = torch.max(torch.min(v, epsilon * torch.ones_like(v)), -epsilon * torch.ones_like(v))

  return v.detach()  #tensor [3,160,160]


def perturb_dataset(model, dataset: TensorDataset, epsilon: float, attack: str, alpha = 0.01, iters = 10): #model, TensorDataset, epsilon (int) 
  perturbed_images = []
  labels = []

  if(attack == 'universal'): 
    v = generate_universal_perturbation(model, dataset, epsilon, alpha, iters) #first generate the universal perturbation
  

  for image, label in dataset:
    celebrity = classes[label]
    if(attack == 'fgsm'):
      perturbed_images.append(perturb_image_fgsm(model, image, celebrity, epsilon))
    elif(attack == 'pgd'): 
      perturbed_images.append(perturb_image_pgd(model, image, celebrity, epsilon, alpha, iters))
    elif(attack == 'universal'): 
       perturbed_images.append(perturb_image_universal(image, v, epsilon))


    label = torch.tensor(label)
    labels.append(label)

  
  perturbed_images = torch.stack(perturbed_images)
  labels = torch.stack(labels)
  return TensorDataset(perturbed_images, labels)








