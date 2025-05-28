import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

import insightface
from insightface.app import FaceAnalysis

SIM_THRESHOLD = 0.5  # or 0.5, depending on what you find works best
TRAINPERCELEB = 80
TESTPERCELEB = 20


def load_arcface_model():
    app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
    return app

def embed_arcface(img_pil):
    """Returns ArcFace embedding from a PIL image."""
    # InsightFace expects numpy BGR images
    img_np = np.array(img_pil)[:, :, ::-1]  # RGB → BGR
    faces = arcface_app.get(img_np)

    if not faces:
        # print(f"[WARN] No faces found")
        # plt.imshow(img_pil)
        # plt.title("No face detected")
        # plt.axis("off")
        # plt.show()
        return None

    return faces[0].embedding  # 512-dim float32

def train_arcface_means(data_dir, celebrities, train_k):
    arc_means = {}
    for celeb in celebrities:
        paths = sorted(glob.glob(os.path.join(data_dir, celeb, "*.*")))[:train_k]
        embs = []
        trainpaths = paths[:TRAINPERCELEB]
        for p in trainpaths:
            img = Image.open(p).convert("RGB")
            emb = embed_arcface(img)
            if emb is not None:
                embs.append(emb)
        if not embs:
            print(f"[WARN] No embeddings found for {celeb}")
            continue
        mean_emb = np.mean(embs, axis=0)
        arc_means[celeb] = mean_emb / np.linalg.norm(mean_emb)
    return arc_means

def test_arcface_transfer(adv_dir, arc_means, celebrities):
    correct = 0
    total = 0

    testpaths = sorted(glob.glob(os.path.join(adv_dir, "*.*")))[TRAINPERCELEB:TRAINPERCELEB + TESTPERCELEB]
    for fname in testpaths:
        img = Image.open(fname).convert("RGB")
        emb = embed_arcface(img)
        if emb is None:
            continue
        emb = emb / np.linalg.norm(emb)
        sims = {c: np.dot(emb, arc_means[c]) for c in celebrities if c in arc_means}
        if not sims:
            continue
        pred = max(sims, key=sims.get)
        pred_score = sims[pred]
        total += 1
        is_correct = (pred == "Brad Pitt") and (pred_score >= SIM_THRESHOLD)
        if is_correct:
            correct += 1

        # if not is_correct:
        #     plt.imshow(img)
        #     plt.title(f"File: {os.path.basename(fname)}\nPredicted: {pred if pred_score >= SIM_THRESHOLD else 'Unknown'} (score: {pred_score:.2f})\nCorrect: Brad Pitt")
        #     plt.axis('off')
        #     plt.pause(1)

        #     print("Similarity scores:")
        #     for celeb in celebrities:
        #         if celeb in sims:
        #             print(f"  {celeb}: {sims[celeb]:.4f}")
        #     print("-" * 40)

    if total == 0:
        acc = 200  # Arbitrary value when no images are tested
        print("\n[ERROR] No images processed. Possibly no faces detected.")
    else:
        acc = correct / total * 100
        print(f"\nArcFace Transfer Accuracy on Brad Pitt Adversarials: {correct}/{total} correct ({acc:.2f}%)\n\n")



if __name__ == "__main__":
    arcface_app = load_arcface_model()
    data_dir = "/Users/derekkirschbaum/cs190a/CS190AProject-s25/Celebrity Faces Dataset"
    adv_dir = "/Users/derekkirschbaum/cs190a/CS190AProject-s25/Bradd-Pert-0.4"
    celebrities = ["Brad Pitt", "Tom Hanks", "Scarlett Johansson", "Megan Fox", "Angelina Jolie"]
    train_k = 5

    #arc_means = train_arcface_means(data_dir, celebrities, train_k)
    if os.path.exists("arcface_means.npy"):
        print("Loading cached arcface class means...")
        arc_means = np.load("arcface_means.npy", allow_pickle=True).item()
    else:
        print("Computing arcface class means...")
        arc_means = train_arcface_means(data_dir, celebrities, train_k)
        np.save("arcface_means.npy", arc_means)

    test_arcface_transfer(adv_dir, arc_means, celebrities)

    data_arcface = np.loadtxt("arcface_accuracy_vs_epsilon.txt")
    eps_arcface, acc_arcface = data_arcface[:, 0], data_arcface[:, 1]

    # Load FGSM accuracy data
    data_fgsm = np.loadtxt("fgsm_accuracy_vs_epsilon.txt")
    eps_fgsm, acc_fgsm = data_fgsm[:, 0], data_fgsm[:, 1]

    # Plot both on the same graph
    plt.plot(eps_arcface, acc_arcface, marker='o', label="ArcFace")
    plt.plot(eps_fgsm, acc_fgsm, marker='s', label="FaceNet")

    # Labeling
    plt.xlabel("Epsilon (ε)")
    plt.ylabel("Accuracy on Brad Pitt (%)")
    plt.title("ArcFace and FaceNet Accuracy vs. Epsilon for FGSM Attacks")
    plt.grid(True)
    plt.legend()
    plt.show()
