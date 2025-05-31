from torch.utils.data import TensorDataset
import torch
from preprocess_data import CLASSES

class Adversary:
    def __init__(self, model, alpha = 0.01, pgd_iters = 10):
        self.model = model
        self.alpha = alpha
        self.pgd_iters = pgd_iters
        self.v = None

    def clamp_eps(self, orig, perturbed, eps):
        upper = orig + eps
        lower = orig - eps
        clipped_high = torch.min(perturbed, upper)
        clipped = torch.max(clipped_high, lower)
        clipped = clipped.clamp(-1.0, 1.0)
        return clipped
    
    def step(self, img, lbl, step_size):
        grad = self.model.compute_gradient(img, CLASSES[lbl])
        return img.clone() + step_size * grad.sign()

    def fgsm(self, img, lbl, eps):
        out = self.step(img, lbl, eps)
        return self.clamp_eps(img, out, eps)

    def pgd(self, img, lbl, eps, iters = None):
        iters = iters or self.pgd_iters
        x = img.clone()
        for _ in range(iters):
            x = self.step(x, lbl, self.alpha)
            x = self.clamp_eps(img, x, eps)
        return x

    def universal(self, img, lbl, eps):
        return self.clamp_eps(img, img + self.v, eps)
    
    def noise(self, img, lbl, eps): 
        rand = torch.rand_like(img) * 2.0 - 1.0
        out = img.clone() + rand
        return self.clamp_eps(img, out, eps)

    def make_universal(self, dataset, eps, alpha = None, iters = None):
        alpha = alpha or self.alpha
        iters = iters or self.pgd_iters
        v = torch.zeros_like(dataset[0][0])
        for _ in range(iters):
            for image, label in dataset:
                image = image.to(torch.float32)
                perturbed_image = image + v
                perturbed_image = self.clamp_eps(image, perturbed_image, eps)

                #Run image through the model
                celebrity = CLASSES[label]
                pred_logits = self.model(perturbed_image.unsqueeze(0))  # batch of 1
                pred_label = torch.argmax(pred_logits, dim=1)

                if pred_label.item() == label: #condition to update v
                    gradient = self.model.compute_gradient(perturbed_image, celebrity) 
                    v = v + (alpha * gradient.sign())
                    v = torch.max(torch.min(v, eps * torch.ones_like(v)), - eps * torch.ones_like(v))

        self.v = v.detach()
        return v.detach()

    def perturb_dataset(self, dataset, eps, attack):
        methods = {
            "fgsm": self.fgsm,
            "pgd":  self.pgd,
            "universal": self.universal,
            "noise": self.noise
        }
        if attack == "universal":
            self.v = self.make_universal(dataset, eps)
        imgs, labs = [], []
        for img, lbl in dataset:
            fn = methods[attack]
            imgs.append(fn(img, lbl, eps))
            lbl = torch.tensor([lbl])
            
            labs.append(lbl)
        
        return TensorDataset(torch.stack(imgs), torch.stack(labs))
