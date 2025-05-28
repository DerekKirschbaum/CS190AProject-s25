import os
import glob
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms
import torch

# ── CONFIG ──────────────────────────────────────────────────────
SCRAPING_DIR = "/Users/derekkirschbaum/cs190a/CS190AProject-s25/ScrapingDataset"
OUTPUT_ROOT = "/Users/derekkirschbaum/cs190a/CS190AProject-s25/just_faces"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── SETUP ────────────────────────────────────────────────────────
mtcnn = MTCNN(image_size=160, margin=0, device=DEVICE)

# ── RUN ──────────────────────────────────────────────────────────
for celeb_name in os.listdir(SCRAPING_DIR):
    celeb_input_dir = os.path.join(SCRAPING_DIR, celeb_name)
    celeb_output_dir = os.path.join(OUTPUT_ROOT, celeb_name.replace(" ", "_") + "_faces")
    os.makedirs(celeb_output_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(celeb_input_dir, "*.*")))

    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            boxes, _ = mtcnn.detect(img)

            if boxes is None or len(boxes) == 0:
                print(f"[{celeb_name} - {i+1}] No face detected in: {os.path.basename(path)}")
                continue

            # Use the first detected face
            x1, y1, x2, y2 = [int(b) for b in boxes[0]]
            face_crop = img.crop((x1, y1, x2, y2))

            out_path = os.path.join(celeb_output_dir, os.path.basename(path))
            face_crop.save(out_path)
            print(f"[{celeb_name} - {i+1}] Saved face: {out_path}")

        except Exception as e:
            print(f"[{celeb_name} - {i+1}] Failed on {path}: {e}")
