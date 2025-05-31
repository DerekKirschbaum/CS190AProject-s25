#imports
from models.simplecnn import SimpleCNN
from models.vgg import VGG
from models.casia import Casia
from perturbations import Adversary
from utils import plot_lines

from perturbations import Adversary
from utils import plot_lines


import matplotlib.pyplot as plt

def evaluate_attack(
    source_model,
    target_models: dict,
    dataset,
    epsilons: list,
    attack_method: str,
    save_path: str
):

    # 1. Create an Adversary object using the source model
    adv = Adversary(source_model)

    # 2. Prepare a place to accumulate accuracies for each target
    accuracies = {name: [] for name in target_models.keys()}

    # 3. For each epsilon, perturb the dataset and evaluate each target model
    for eps in epsilons:
        perturbed_dataset = adv.perturb_dataset(dataset, eps, attack_method)
        for model_name, model_obj in target_models.items():
            acc = model_obj.compute_accuracy(perturbed_dataset)
            accuracies[model_name].append(acc)

    # 4. Gather lists of accuracies in the same order as target_models.keys()
    model_names = list(target_models.keys())
    accuracy_lists = [accuracies[name] for name in model_names]

    # 5. Build plot metadata
    title = f"{attack_method.upper()} Attack: Accuracy vs Epsilon"
    xlabel = "Epsilon"
    ylabel = "Accuracy"
    labels = model_names

    # 6. Use plot_lines to create and save the figure
    #    plot_lines will save to save_path + title (without file extension),
    #    so we'll append ".png" explicitly in save_path if desired.
    plot_lines(
        x=epsilons,
        ys=accuracy_lists,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        save_path=save_path,
        labels=labels,
        marker='o'
    )



from preprocess_data import TEST_SET

figure_path = './figures/'

if __name__ == "__main__":
    cnn = SimpleCNN()
    vgg = VGG()
    casia = Casia()


    cnn_path = "./checkpoints/simplecnn.npy"
    vgg_path = "./checkpoints/vgg.npy"
    casia_path = "./checkpoints/casia.npy"

    cnn.load(cnn_path)
    vgg.load(vgg_path)
    casia.load(casia_path)


    # 2. Prepare the list of target models and their labels
    target_models = [cnn, vgg, casia]
    model_labels  = ["CNN", "VGG", "Casia"]

    # 3. Define the epsilons to test
    epsilons = [round(i * 0.05, 2) for i in range(1)]  # [0.0, 0.05, 0.10, …, 0.45]

    # 4. Call evaluate_attack, using `cnn` as the source for crafting perturbations
    evaluate_attack(
        source_model=cnn,
        target_models=target_models,
        model_labels=model_labels,
        dataset=TEST_SET,
        epsilons=epsilons,
        attack_method="fgsm",
        save_path=figure_path
    )

