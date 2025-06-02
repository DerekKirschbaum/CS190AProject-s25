from functions.perturbations import Adversary
from models.vgg import VGG
from models.simplecnn import SimpleCNN
from models.arcface import ArcFace
from preprocess_data import TEST_SET
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

vgg_path = './checkpoints/vgg.npy'
cnn_path = './checkpoints/simplecnn.npy'
arcface_path = "./checkpoints/arcface.npy"

if __name__ == "__main__":
    # vgg_model = VGG()
    # cnn_model = SimpleCNN()
    # arcface_model = ArcFace()

    # cnn_model.load(cnn_path)
    # vgg_model.load(vgg_path)
    # arcface_model.load(arcface_path)

    # vgg_accuracy = vgg_model.compute_accuracy(TEST_SET)
    # cnn_accuracy = cnn_model.compute_accuracy(TEST_SET)
    # arcface_accuracy = arcface_model.compute_accuracy(TEST_SET)

    # print("vgg baseline accuracy: ", vgg_accuracy)
    # print("cnn baseline accuracy: ", cnn_accuracy)
    # print("arc baseline accuracy: ", arcface_accuracy)

    # epsilons = np.arange(0.00, 0.55, 0.05)

    # acc_arc_on_vgg = []
    # acc_arc_on_cnn = []
    # acc_arc_on_vgg_cos = []
    # acc_arc_on_cnn_cos = []

    # for epsilon in epsilons:
    #     print(f"\nEvaluating for epsilon = {epsilon:.2f}...")

    #     vgg_adv = Adversary(vgg_model)
    #     cnn_adv = Adversary(cnn_model)

    #     vgg_perturbed_set = vgg_adv.perturb_dataset(TEST_SET, eps=epsilon, attack='fgsm')
    #     cnn_perturbed_set = cnn_adv.perturb_dataset(TEST_SET, eps=epsilon, attack='fgsm')

    #     acc_arc_vgg = arcface_model.compute_accuracy(vgg_perturbed_set)
    #     acc_arc_cnn = arcface_model.compute_accuracy(cnn_perturbed_set)

    #     acc_arc_vgg_cos = arcface_model.compute_accuracy_with_cos(vgg_perturbed_set, 0.5)
    #     acc_arc_cnn_cos = arcface_model.compute_accuracy_with_cos(cnn_perturbed_set, 0.5)


    #     acc_arc_on_vgg.append(acc_arc_vgg)
    #     acc_arc_on_cnn.append(acc_arc_cnn)
    #     acc_arc_on_vgg_cos.append(acc_arc_vgg_cos)
    #     acc_arc_on_cnn_cos.append(acc_arc_cnn_cos)


    #     print(f"  ARC on VGG: {acc_arc_vgg:.2f}")
    #     print(f"  ARC on CNN: {acc_arc_cnn:.2f}")
    #     print(f"  ARC on VGG with cos: {acc_arc_vgg_cos:.2f}")
    #     print(f"  ARC on CNN with cos: {acc_arc_cnn_cos:.2f}")

    epsilons = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    # First 4 accuracy arrays
    acc_vgg_on_vgg = [99.31, 14.24, 3.47, 1.04, 1.74, 3.82, 6.60, 9.03, 12.15, 10.76, 15.62]
    acc_vgg_on_cnn = [99.31, 98.96, 98.61, 97.22, 95.14, 91.32, 86.46, 79.51, 73.26, 65.62, 56.60]
    acc_cnn_on_cnn = [68.75, 29.86, 11.81, 3.47, 1.04, 0.35, 0.00, 0.00, 0.00, 0.00, 0.00]
    acc_cnn_on_vgg = [68.75, 67.36, 64.93, 64.24, 63.19, 63.19, 61.81, 57.99, 55.21, 54.17, 52.78]

    # New 4 accuracy arrays
    acc_arc_on_vgg     = [97.92, 95.14, 93.06, 86.81, 82.64, 74.31, 65.97, 60.42, 53.47, 46.88, 42.71]
    acc_arc_on_cnn     = [97.92, 97.92, 97.22, 97.22, 95.83, 92.01, 86.11, 81.94, 71.18, 60.76, 51.39]
    acc_arc_on_vgg_cos = [73.96, 38.89, 17.36, 8.33, 4.17, 1.04, 0.00, 0.00, 0.00, 0.00, 0.00]
    acc_arc_on_cnn_cos = [73.96, 70.14, 62.15, 42.71, 20.49, 9.03, 2.43, 0.69, 0.35, 0.00, 0.00]

    # Plot
    plt.figure(figsize=(13, 7))

    # Plot each accuracy curve
    plt.plot(epsilons, acc_vgg_on_vgg, marker='o', label='VGG on VGG')
    plt.plot(epsilons, acc_vgg_on_cnn, marker='o', label='VGG on CNN')
    plt.plot(epsilons, acc_cnn_on_cnn, marker='o', label='CNN on CNN')
    plt.plot(epsilons, acc_cnn_on_vgg, marker='o', label='CNN on VGG')

    plt.plot(epsilons, acc_arc_on_vgg, marker='s', linestyle='--', label='ArcFace on VGG')
    plt.plot(epsilons, acc_arc_on_cnn, marker='s', linestyle='--', label='ArcFace on CNN')
    plt.plot(epsilons, acc_arc_on_vgg_cos, marker='^', linestyle=':', label='ArcFace on VGG (cos)')
    plt.plot(epsilons, acc_arc_on_cnn_cos, marker='^', linestyle=':', label='ArcFace on CNN (cos)')

    # Labels and legend
    plt.xlabel("Epsilon (ε)", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.title("Model Accuracy vs Epsilon", fontsize=16)
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


    # # Plot the results
    # plt.figure(figsize=(10, 6))
    # plt.plot(epsilons, acc_vgg_on_vgg, label='VGG on VGG', marker='o')
    # plt.plot(epsilons, acc_vgg_on_cnn, label='VGG on CNN', marker='o')
    # plt.plot(epsilons, acc_cnn_on_cnn, label='CNN on CNN', marker='o')
    # plt.plot(epsilons, acc_cnn_on_vgg, label='CNN on VGG', marker='o')

    # plt.xlabel('Epsilon (ε)')
    # plt.ylabel('Accuracy (%)')
    # plt.title('Adversarial Transfer Accuracy vs. Epsilon')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("adversarial_accuracy_plot.png")  # Optional: save to file
    # plt.show()


# Results because this takes like 10 minutes to run :)

# vgg baseline accuracy:  99.30555555555556
# cnn baseline accuracy:  68.75

# Evaluating for epsilon = 0.00...
#   VGG on VGG: 99.31
#   VGG on CNN: 99.31
#   CNN on CNN: 68.75
#   CNN on VGG: 68.75

# Evaluating for epsilon = 0.05...
#   VGG on VGG: 14.24
#   VGG on CNN: 98.96
#   CNN on CNN: 29.86
#   CNN on VGG: 67.36

# Evaluating for epsilon = 0.10...
#   VGG on VGG: 3.47
#   VGG on CNN: 98.61
#   CNN on CNN: 11.81
#   CNN on VGG: 64.93

# Evaluating for epsilon = 0.15...
#   VGG on VGG: 1.04
#   VGG on CNN: 97.22
#   CNN on CNN: 3.47
#   CNN on VGG: 64.24

# Evaluating for epsilon = 0.20...
#   VGG on VGG: 1.74
#   VGG on CNN: 95.14
#   CNN on CNN: 1.04
#   CNN on VGG: 63.19

# Evaluating for epsilon = 0.25...
#   VGG on VGG: 3.82
#   VGG on CNN: 91.32
#   CNN on CNN: 0.35
#   CNN on VGG: 63.19

# Evaluating for epsilon = 0.30...
#   VGG on VGG: 6.60
#   VGG on CNN: 86.46
#   CNN on CNN: 0.00
#   CNN on VGG: 61.81

# Evaluating for epsilon = 0.35...
#   VGG on VGG: 9.03
#   VGG on CNN: 79.51
#   CNN on CNN: 0.00
#   CNN on VGG: 57.99

# Evaluating for epsilon = 0.40...
#   VGG on VGG: 12.15
#   VGG on CNN: 73.26
#   CNN on CNN: 0.00
#   CNN on VGG: 55.21

# Evaluating for epsilon = 0.45...
#   VGG on VGG: 10.76
#   VGG on CNN: 65.62
#   CNN on CNN: 0.00
#   CNN on VGG: 54.17

# Evaluating for epsilon = 0.50...
#   VGG on VGG: 15.62
#   VGG on CNN: 56.60
#   CNN on CNN: 0.00
#   CNN on VGG: 52.78


# vgg baseline accuracy:  99.30555555555556
# cnn baseline accuracy:  68.75
# arc baseline accuracy:  97.91666666666666

# Evaluating for epsilon = 0.00...
#   ARC on VGG: 97.92
#   ARC on CNN: 97.92
#   ARC on VGG with cos: 73.96
#   ARC on CNN with cos: 73.96

# Evaluating for epsilon = 0.05...
#   ARC on VGG: 95.14
#   ARC on CNN: 97.92
#   ARC on VGG with cos: 38.89
#   ARC on CNN with cos: 70.14

# Evaluating for epsilon = 0.10...
#   ARC on VGG: 93.06
#   ARC on CNN: 97.22
#   ARC on VGG with cos: 17.36
#   ARC on CNN with cos: 62.15

# Evaluating for epsilon = 0.15...
#   ARC on VGG: 86.81
#   ARC on CNN: 97.22
#   ARC on VGG with cos: 8.33
#   ARC on CNN with cos: 42.71

# Evaluating for epsilon = 0.20...
#   ARC on VGG: 82.64
#   ARC on CNN: 95.83
#   ARC on VGG with cos: 4.17
#   ARC on CNN with cos: 20.49

# Evaluating for epsilon = 0.25...
#   ARC on VGG: 74.31
#   ARC on CNN: 92.01
#   ARC on VGG with cos: 1.04
#   ARC on CNN with cos: 9.03

# Evaluating for epsilon = 0.30...
#   ARC on VGG: 65.97
#   ARC on CNN: 86.11
#   ARC on VGG with cos: 0.00
#   ARC on CNN with cos: 2.43

# Evaluating for epsilon = 0.35...
#   ARC on VGG: 60.42
#   ARC on CNN: 81.94
#   ARC on VGG with cos: 0.00
#   ARC on CNN with cos: 0.69

# Evaluating for epsilon = 0.40...
#   ARC on VGG: 53.47
#   ARC on CNN: 71.18
#   ARC on VGG with cos: 0.00
#   ARC on CNN with cos: 0.35

# Evaluating for epsilon = 0.45...
#   ARC on VGG: 46.88
#   ARC on CNN: 60.76
#   ARC on VGG with cos: 0.00
#   ARC on CNN with cos: 0.00

# Evaluating for epsilon = 0.50...
#   ARC on VGG: 42.71
#   ARC on CNN: 51.39
#   ARC on VGG with cos: 0.00
#   ARC on CNN with cos: 0.00