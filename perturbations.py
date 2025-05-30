from torch.utils.data import TensorDataset
from typing import List
import torch
from datasets import classes

class Adversary:
    def __init__(self, model, classes: List[str], alpha=0.01, pgd_iters=10):
        self.model = model
        self.classes = classes
        self.alpha = alpha
        self.pgd_iters = pgd_iters

    def clamp_eps(self, orig, perturbed, eps):
        clipped = torch.max(torch.min(perturbed, orig + eps), orig - eps)
        return clipped.clamp(-1.0, 1.0)

    def step(self, img, lbl, step_size):
        grad = self.model.compute_gradient(img, self.classes[lbl])
        return img + step_size * grad.sign()

    def fgsm(self, img, lbl, eps):
        out = self.step(img, lbl, eps)
        return self.clamp_eps(img, out, eps)

    def pgd(self, img, lbl, eps, iters=None):
        iters = iters or self.pgd_iters
        x = img.clone()
        for _ in range(iters):
            x = self.step(x, lbl, self.alpha)
            x = self.clamp_eps(img, x, eps)
        return x

    def universal(self, img, v, eps):
        return self.clamp_eps(img, img + v, eps)

    def make_universal(self, dataset: TensorDataset, eps, alpha=None, iters=None):
        alpha = alpha or self.alpha
        iters = iters or self.pgd_iters
        v = torch.zeros_like(dataset[0][0])
        for _ in range(iters):
            for image, label in dataset:
                image = image.to(torch.float32)
                perturbed_image = image + v
                perturbed_image = self.clamp_eps(image, perturbed_image, eps)

                #Run image through the model
                celebrity = classes[label]
                pred_logits = self.model(perturbed_image.unsqueeze(0))  # batch of 1
                pred_label = torch.argmax(pred_logits, dim=1)

                if pred_label.item() == label: #condition to update v
                    gradient = self.model.compute_gradient(perturbed_image, celebrity) 
                    v = v + (alpha * gradient.sign())
                    v = torch.max(torch.min(v, eps * torch.ones_like(v)), - eps * torch.ones_like(v))

        return v.detach()

    def perturb_dataset(
        self,
        dataset: TensorDataset,
        eps: float,
        attack: str
    ) -> TensorDataset:
        methods = {
            "fgsm": self.fgsm,
            "pgd":  self.pgd,
            "universal": lambda img, lbl: self.universal(img, v, eps)
        }
        if attack == "universal":
            v = self.make_universal(dataset, eps)
        imgs, labs = [], []
        for img, lbl in dataset:
            fn = methods[attack]
            imgs.append(fn(img, lbl))
            labs.append(lbl)
        return TensorDataset(torch.stack(imgs), torch.stack(labs))
