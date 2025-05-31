# imports
from models.simplecnn import SimpleCNN
from models.vgg import VGG
from models.casia import Casia
from perturbations import Adversary
from utils import plot_lines

from preprocess_data import TEST_SET

figure_path = './figures/'

def evaluate_attack(
    source_model,
    target_models: list,
    model_labels: list,
    dataset,
    epsilons: list,
    attack_method: str,
    save_path: str
):

    # 1. Create an Adversary object using the source model
    adv = Adversary(source_model)

    # 2. Prepare a place to accumulate accuracies for each target
    accuracies = {label: [] for label in model_labels}

    # 3. For each epsilon, perturb the dataset and evaluate each target model
    for eps in epsilons:
        perturbed_dataset = adv.perturb_dataset(dataset, eps, attack_method)
        for model_obj, label in zip(target_models, model_labels):
            acc = model_obj.compute_accuracy(perturbed_dataset)
            accuracies[label].append(acc)

    # 4. Gather accuracy lists in the same order as model_labels
    accuracy_lists = [accuracies[label] for label in model_labels]

    # 5. Build plot metadata
    title = f"{attack_method.upper()} Attack (Source: {source_model.__class__.__name__}): Accuracy vs Epsilon"
    xlabel = "Epsilon"
    ylabel = "Accuracy %"
    labels = model_labels

    # 6. Use plot_lines to create and save the figure
    #    plot_lines will save to save_path + title (without extension),
    #    so append ".png" to save_path+title when saving on disk.
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


if __name__ == "__main__":
    # Instantiate and load each model
    cnn = SimpleCNN()
    vgg = VGG()
    casia = Casia()

    cnn_path = "./checkpoints/simplecnn.npy"
    vgg_path = "./checkpoints/vgg.npy"
    casia_path = "./checkpoints/casia.npy"

    cnn.load(cnn_path)
    vgg.load(vgg_path)
    casia.load(casia_path)

    # Prepare the list of target models and their labels
    target_models = [cnn, vgg, casia]
    model_labels  = ["CNN", "VGG", "Casia"]

    # Define the epsilons to test
    epsilons = [round(i * 0.05, 2) for i in range(10)]  # [0.0, 0.05, 0.10, ..., 0.45]

    # Call evaluate_attack, using `cnn` as the source for crafting perturbations
    evaluate_attack(
        source_model=vgg,
        target_models=target_models,
        model_labels=model_labels,
        dataset=TEST_SET,
        epsilons=epsilons,
        attack_method="fgsm",
        save_path=figure_path
    )

    evaluate_attack(
        source_model=vgg,
        target_models=target_models,
        model_labels=model_labels,
        dataset=TEST_SET,
        epsilons=epsilons,
        attack_method="pgd",
        save_path=figure_path
    )

    evaluate_attack(
        source_model=vgg,
        target_models=target_models,
        model_labels=model_labels,
        dataset=TEST_SET,
        epsilons=epsilons,
        attack_method="noise",
        save_path=figure_path
    )

    evaluate_attack(
        source_model=vgg,
        target_models=target_models,
        model_labels=model_labels,
        dataset=TEST_SET,
        epsilons=epsilons,
        attack_method="universal",
        save_path=figure_path
    )



