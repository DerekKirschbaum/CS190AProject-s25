import os
import matplotlib.pyplot as plt
from perturbations.utils import plot_lines

epsilons = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]

epsilons = [0.00, 0.05, 0.15, 0.25]

cos_accuracies_linear_source = [
    [74.31, 24.31, 0.35, 0.00],   # Target: Linear
    [65.28, 43.40, 8.33, 1.04],   # Target: CNN
    [99.31, 99.31, 96.53, 89.93], # Target: VGG
]

labels = ['Linear', 'SimpleCNN', 'VGG']
plot_lines(
    x=epsilons,
    ys=cos_accuracies_linear_source,
    title="FGSM Attack (Source: Linear): Cos Accuracy vs Epsilon",
    xlabel="Epsilon",
    ylabel="Cos Accuracy (%)",
    save_path="./figures/cnnthresh",
    labels=labels,
    marker='o'
)

cos_accuracies_cnn_source = [
    [74.31, 60.07, 29.51, 8.33],   # Target: Linear
    [65.28, 25.35, 4.17, 0.00],    # Target: CNN
    [99.31, 99.31, 98.26, 96.53],  # Target: VGG
]

plot_lines(
    x=epsilons,
    ys=cos_accuracies_cnn_source,
    title="FGSM Attack (Source: SimpleCNN): Cos Accuracy vs Epsilon",
    xlabel="Epsilon",
    ylabel="Cos Accuracy (%)",
    save_path="./figures/cnnthresh",
    labels=labels,
    marker='o'
)

