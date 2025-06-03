import os
import matplotlib.pyplot as plt
from utils import plot_lines

epsilons = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]

reg_accuracies = [
    [70.83, 99.31, 99.31, 97.92, 79.17],  # epsilon = 0.00
    [70.49, 99.31, 99.65, 97.92, 9.72],   # epsilon = 0.05
    [70.49, 98.61, 98.96, 98.26, 10.76],  # epsilon = 0.10
    [70.49, 97.57, 94.10, 97.92, 13.54],  # epsilon = 0.15
    [69.10, 95.14, 89.93, 97.22, 14.58],  # epsilon = 0.20
    [67.36, 92.01, 81.25, 94.44, 20.14],  # epsilon = 0.25
]

target_model_names = ["CNN", "VGG", "Casia", "ArcFace", "VIT"]

# Transpose so each inner list corresponds to a model across epsilons
reg_accuracies_per_model = list(map(list, zip(*reg_accuracies)))

# Make sure the save directory exists
save_dir = "./figures/vittests2"
os.makedirs(save_dir, exist_ok=True)

# Plot
plot_lines(
    x=epsilons,
    ys=reg_accuracies_per_model,
    title="FGSM Attack (Source: ViTEmbedder): Accuracy vs Epsilon",
    xlabel="Epsilon",
    ylabel="Regular Accuracy (%)",
    save_path=save_dir,
    labels=target_model_names,
    marker='o'
)