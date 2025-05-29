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
DATA_DIR   = "./resized_faces"
CELEBS     = ["Brad_Pitt_faces", "Tom_Hanks_faces", "Scarlett_Johansson_faces", "Megan_Fox_faces", "Angelina_Jolie_faces"]
TRAIN_K    = 250
TEST_K     = 100
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# ── MAIN ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    means = np.load("class_means.npy", allow_pickle=True).item()
    test_model("Facenet", DATA_DIR, CELEBS, means)

