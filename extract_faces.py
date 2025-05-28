import os
import glob
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms

import torch

# ── CONFIG ──────────────────────────────────────────────────────
INPUT_DIR = "/Users/derekkirschbaum/cs190a/CS190AProject-s25/Celebrity Faces Dataset/Angelina Jolie"   # Change this to your 100-image folder
OUTPUT_DIR = "/Users/derekkirschbaum/cs190a/CS190AProject-s25/just_faces/Angelina_Jolie_faces"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── SETUP ────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
mtcnn = MTCNN(image_size=160, margin=0, device=DEVICE)

# ── RUN ──────────────────────────────────────────────────────────
image_paths = sorted(glob.glob(os.path.join(INPUT_DIR, "*.*")))

for i, path in enumerate(image_paths):
    try:
        img = Image.open(path).convert("RGB")
        boxes, _ = mtcnn.detect(img)

        if boxes is None or len(boxes) == 0:
            print(f"[{i+1}] No face detected in: {os.path.basename(path)}")
            continue

        # Use the first detected face
        x1, y1, x2, y2 = [int(b) for b in boxes[0]]
        face_crop = img.crop((x1, y1, x2, y2))

        # Save the cropped face
        out_path = os.path.join(OUTPUT_DIR, os.path.basename(path))
        face_crop.save(out_path)
        print(f"[{i+1}] Saved face: {out_path}")

    except Exception as e:
        print(f"[{i+1}] Failed on {path}: {e}")