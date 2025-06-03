from torch.utils.data import TensorDataset
import torch
from preprocess_data import CLASSES
from perturbations.utils import plot_lines

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
        self.alpha = eps/10
        delta = torch.empty_like(img).uniform_(-eps, eps)
        x = self.clamp_eps(img, img + delta, eps)
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
        alpha = eps/10
        iters = iters or self.pgd_iters
        v = torch.zeros_like(dataset[0][0])
        for _ in range(iters):
            for image, label in dataset:
                image = image.to(torch.float32)
                perturbed_image = image + v
                perturbed_image = self.clamp_eps(image, perturbed_image, eps)

                #Run image through the model
                celebrity = CLASSES[label]
                pred_logits = self.model.forward(perturbed_image.unsqueeze(0))  # batch of 1
                
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



def evaluate_attack(
    source_model,
    target_models: list,
    model_labels: list,
    dataset,
    epsilons: list,
    attack_method: str,
    save_path: str,
    threshold: float = 0.5
):
    print(f"\n=== Evaluating {attack_method.upper()} Attack ===")
    print(f"Source Model: {source_model.__class__.__name__}")
    print("Target Models:", [m.__class__.__name__ for m in target_models])
    # 1. Create an Adversary object using the source model
    adv = Adversary(source_model)

    # 2. Prepare a place to accumulate accuracies for each target
    accuracies_reg = {label: [] for label in model_labels}
    accuracies_cos = {label: [] for label in model_labels}

    # 3. For each epsilon, perturb the dataset and evaluate each target model
    for eps in epsilons:
        print(f"\n--- Epsilon = {eps:.2f} ---")
        perturbed_dataset = adv.perturb_dataset(dataset, eps, attack_method)
        for model_obj, label in zip(target_models, model_labels):
            acc_reg, acc_cos = model_obj.compute_accuracy_with_cos(perturbed_dataset, threshold)
            
            print(f"Reg Accuracy: Target: {label} | Accuracy: {acc_reg:.2f}%")
            print(f"Cos Accuracy: Target: {label} | Accuracy: {acc_cos:.2f}%")
            accuracies_reg[label].append(acc_reg)
            accuracies_cos[label].append(acc_cos)

    # 4. Gather accuracy lists in the same order as model_labels
    accuracy_lists_reg = [accuracies_reg[label] for label in model_labels]
    accuracy_lists_cos = [accuracies_cos[label] for label in model_labels]



    # 5. Build plot metadata
    title_reg = f"{attack_method.upper()} Attack (Source: {source_model.__class__.__name__}): Nearest Class Accuracy vs Epsilon,"
    title_cos = f"{attack_method.upper()} Attack (Source: {source_model.__class__.__name__}): Threshold Class Accuracy vs Epsilon"
    xlabel = "Epsilon"
    ylabel_reg = "Nearest Class Accuracy %"
    ylabel_cos = "Threshold Class Accuracy % "
    labels = model_labels

    # 6. Use plot_lines to create and save the figure
    #    plot_lines will save to save_path + title (without extension),
    #    so append ".png" to save_path+title when saving on disk.
    plot_lines(
        x=epsilons,
        ys=accuracy_lists_reg,
        title=title_reg,
        xlabel=xlabel,
        ylabel=ylabel_reg,
        save_path=save_path,
        labels=labels,
        marker='o'
    )
    plot_lines(
        x=epsilons,
        ys=accuracy_lists_cos,
        title=title_cos,
        xlabel=xlabel,
        ylabel=ylabel_cos,
        save_path=save_path,
        labels=labels,
        marker='o'
    )