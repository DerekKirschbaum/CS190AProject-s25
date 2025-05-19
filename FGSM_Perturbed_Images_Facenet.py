import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# ── CONFIG ────────────────────────────────────────────────────────────────────────
DATA_DIR   = "./Celebrity Faces Dataset"
CELEBS     = ["Brad Pitt", "Tom Hanks", "Scarlett Johansson", "Megan Fox", "Angelina Jolie"]
TRAIN_K    = 80
TEST_K     = 20
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPSILON    = 0.06   # max per-pixel change on [-1,1] scale

# ── SET UP MTCNN & FaceNet ───────────────────────────────────────────────────────
mtcnn = MTCNN(image_size=160, margin=0, device=DEVICE)
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
to_tensor = transforms.ToTensor()

def load_face_tensor(path):
    """
    Returns a torch tensor (3×160×160) in [-1,1].
    If MTCNN fails, falls back to resizing the full image.
    """
    img = Image.open(path).convert('RGB')
    face = mtcnn(img)  # tries detect+align → [3×160×160] in [0,1]
    if face is None:
        face = to_tensor(img.resize((160,160)))
    return face.mul(2.).sub(1.)  # scale [0,1]→[-1,1]

def embed_tensor(face_tensor):
    """512-d L2-normalized embedding as a numpy array."""
    with torch.no_grad():
        emb = model(face_tensor.unsqueeze(0).to(DEVICE))
        emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()[0]

# ── ORIGINAL TRAIN / TEST ───────────────────────────────────────────────────────
def train_model(model_name: str, data_dir: str, celebrities: list, train_per_celeb: int = TRAIN_K):
    class_means = {}
    for celeb in celebrities:
        img_paths = sorted(glob.glob(os.path.join(data_dir, celeb, '*.*')))
        train_paths = img_paths[:train_per_celeb]
        embs = [embed_tensor(load_face_tensor(p)) for p in train_paths]
        mean_emb = np.mean(embs, axis=0)
        class_means[celeb] = mean_emb / np.linalg.norm(mean_emb)
    return class_means

def test_model(model_name: str,
               data_dir: str,
               celebrities: list,
               class_means: dict,
               train_per_celeb: int = TRAIN_K,
               test_per_celeb: int = TEST_K):
    total = 0
    correct = 0
    correct_per_celeb = {c: 0 for c in celebrities}

    for celeb in celebrities:
        img_paths = sorted(glob.glob(os.path.join(data_dir, celeb, '*.*')))
        test_paths = img_paths[train_per_celeb:train_per_celeb + test_per_celeb]
        for p in test_paths:
            emb = embed_tensor(load_face_tensor(p))
            emb = emb / np.linalg.norm(emb)
            sims = {c: np.dot(emb, class_means[c]) for c in celebrities}
            pred = max(sims, key=sims.get)
            total += 1
            if pred == celeb:
                correct += 1
                correct_per_celeb[celeb] += 1

    overall_acc = correct / total * 100
    print(f"\nBaseline Facenet accuracy: {overall_acc:.2f}%")
    for celeb in celebrities:
        acc = correct_per_celeb[celeb] / test_per_celeb * 100
        print(f"{celeb:15s} accuracy: {acc:.2f}%")

# ── SINGLE-STEP FGSM (classification loss) ───────────────────────────────────────

def generate_FGSM(face_tensor, class_means, celebrities, eps=EPSILON):
    """
    Single-step FGSM: perturb face_tensor to *maximize* the cross-entropy
    loss on the five-way “logits” = embedding·class_mean.
    Returns (adv_tensor, delta_tensor).
    """
    # 1) build a stacked tensor of all class‐means: shape (512,5)
    cm_torch = torch.stack([torch.tensor(class_means[c], device=DEVICE, dtype=torch.float32)
                            for c in celebrities], dim=1)

    # 2) prepare input for gradient
    x = face_tensor.unsqueeze(0).clone().detach().requires_grad_(True).to(DEVICE)  # (1,3,160,160)

    # 3) forward → embedding → normalize → compute 5 “logits”
    emb = model(x)                                          # (1,512)
    embn = F.normalize(emb, p=2, dim=1)                     # (1,512)
    logits = embn @ cm_torch                                # (1,5)

    # 4) true label index for Brad Pitt
    y_true = torch.tensor([celebrities.index("Brad Pitt")], device=DEVICE)

    # 5) cross-entropy loss (we want to *increase* this)
    loss = F.cross_entropy(logits, y_true)
    loss.backward()

    # 6) FGSM step: move *with* the gradient sign to increase loss
    delta = eps * x.grad.sign()                             # (1,3,160,160)
    adv   = torch.clamp(x + delta, -1.0, 1.0)               # (1,3,160,160)

    return adv.squeeze(0).detach(), delta.squeeze(0).detach()


# ── TEST FGSM ────────────────────────────────────────────────────────────────────

def test_FGSM(model_name, data_dir, celebrities, class_means,
              train_per_celeb=TRAIN_K, test_per_celeb=TEST_K, eps=EPSILON):
    """
    Same as your original test_model, but for Brad Pitt we first
    call generate_FGSM → get (adv,delta) → plot once → classify adv.
    """
    output_file = "fgsm_accuracy_vs_epsilon.txt"
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():  # skip empty lines
                    saved_eps, saved_acc = line.strip().split()
                    if float(saved_eps) == eps:
                        print(f"\n[Skipping] ε={eps} already evaluated. Brad Pitt accuracy: {saved_acc}%")
                        return
                    
    total, correct = 0, 0
    correct_per_celeb = {c:0 for c in celebrities}
    plotted = False

    for celeb in celebrities:
        paths = sorted(glob.glob(os.path.join(data_dir, celeb, '*.*')))
        test_paths = paths[train_per_celeb:train_per_celeb+test_per_celeb]
        for p in test_paths:
            ft = load_face_tensor(p)  # your existing loader

            if celeb == "Brad Pitt":
                adv, delta = generate_FGSM(ft, class_means, celebrities, eps)

                if not plotted:
                    orig_np = ((ft.cpu().numpy().transpose(1,2,0) + 1) / 2)
                    dp      = delta.cpu().numpy().transpose(1,2,0)
                    dp      = (dp - dp.min())/(dp.max()-dp.min())
                    adv_np  = ((adv.cpu().numpy().transpose(1,2,0) + 1) / 2)

                    plt.figure(); plt.imshow(orig_np); plt.title("Original");    plt.axis('off')
                    plt.figure(); plt.imshow(dp);      plt.title("Perturbation");plt.axis('off')
                    plt.figure(); plt.imshow(adv_np);  plt.title("Adversarial"); plt.axis('off')
                    plt.show()
                    plotted = True

                inp = adv
            else:
                inp = ft

            emb_adv = embed_tensor(inp)   # your existing embed → numpy
            sims    = {c: np.dot(emb_adv, class_means[c]) for c in celebrities}
            pred    = max(sims, key=sims.get)

            total += 1
            if pred == celeb:
                correct += 1
                correct_per_celeb[celeb] += 1

    overall_acc = correct/total * 100
    print(f"\nFGSM (ε={eps}) accuracy:")
    print(f" Overall: {overall_acc:.2f}%")
    for c in celebrities:
        acc = correct_per_celeb[c] / test_per_celeb * 100
        print(f" {c:15s}: {acc:.2f}%")
    
    brad_acc = correct_per_celeb["Brad Pitt"] / test_per_celeb * 100

    with open(output_file, "a") as f:
        f.write(f"{eps}\t{brad_acc:.2f}\n")
    
    with open(output_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        lines = sorted(lines, key=lambda x: float(x.split()[0]))

    with open(output_file, "w") as f:
        for line in lines:
            f.write(line + "\n")

# ── SAVE 100 BRAD PERTURBATIONS ──────────────────────────────────────────────────
def save_all_bradd_perturbations(class_means, celebrities, eps=EPSILON):
    out_dir = os.path.join(".", f"Bradd-Pert-{eps}")
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(out_dir) and len(glob.glob(os.path.join(out_dir, "*.*"))) >= 100:
        print(f"Adversarial images already exist in {out_dir}, skipping generation.")
        return

    # take first 100 Brad Pitt images
    brad_paths = sorted(glob.glob(os.path.join(DATA_DIR, "Brad Pitt", "*.*")))[:100]
    for p in brad_paths:
        # 1) load full-res
        img = Image.open(p).convert("RGB")

        # 2) detect face bounding box
        boxes, _ = mtcnn.detect(img)
        if boxes is None or len(boxes)==0:
            # if no face found, perturb whole image
            x1, y1, x2, y2 = 0, 0, img.width, img.height
        else:
            x1, y1, x2, y2 = [int(b) for b in boxes[0]]

        # 3) crop that rectangle (no alignment)
        crop = img.crop((x1, y1, x2, y2))

        # 4) make a 160×160 tensor in [-1,1]
        face_crop = to_tensor(crop.resize((160,160))).mul(2.).sub(1.)

        # 5) generate FGSM on that patch
        adv_tensor, _ = generate_FGSM(face_crop, class_means, celebrities, eps)

        # 6) convert back to [0,255] uint8 and resize to original box
        adv_np = ((adv_tensor.cpu().numpy().transpose(1,2,0) + 1)/2 * 255).astype(np.uint8)
        adv_patch = Image.fromarray(adv_np).resize((x2-x1, y2-y1))

        # 7) paste the adversarial patch over the original
        out = img.copy()
        out.paste(adv_patch, (x1, y1))

        # 8) save with same resolution
        out.save(os.path.join(out_dir, os.path.basename(p)))


# ── MAIN ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RETRAIN = False

    if not RETRAIN and os.path.exists("class_means.npy"):
        print("Loading cached class means...")
        means = np.load("class_means.npy", allow_pickle=True).item()
    else:
        print("Training model and computing class means...")
        means = train_model("Facenet", DATA_DIR, CELEBS)
        np.save("class_means.npy", means)

    # test_model("Facenet", DATA_DIR, CELEBS, means)
    test_FGSM("Facenet", DATA_DIR, CELEBS, means, eps=EPSILON)
    save_all_bradd_perturbations(means, CELEBS, eps=EPSILON)

    data = np.loadtxt("fgsm_accuracy_vs_epsilon.txt")
    epsilons, accuracies = data[:, 0], data[:, 1]

    plt.plot(epsilons, accuracies, marker='o')
    plt.xlabel("Epsilon (ε)")
    plt.ylabel("Accuracy on Brad Pitt (%)")
    plt.title("FGSM Attack: Brad Pitt Accuracy vs. Epsilon")
    plt.grid(True)
    plt.show()


