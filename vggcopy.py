import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from dataset import classes, DATA_DIR



CELEBS = classes
TRAIN_K    = 250
TEST_K     = 100
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VGGModel(): 
    def __init__(): 
        # ── SET UP MTCNN & FaceNet ───────────────────────────────────────────────────────
        mtcnn = MTCNN(image_size=160, margin=0, device=DEVICE)
        model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
        to_tensor = transforms.ToTensor()


    def load_face_tensor(self,path):
        """
        Returns a torch tensor (3×160×160) in [-1,1].
        If MTCNN fails, falls back to resizing the full image.
        """
        img = Image.open(path).convert('RGB')
        face = self.mtcnn(img)  # tries detect+align → [3×160×160] in [0,1]
        if face is None:
            face = torch.Tensor(img.resize((160,160)))
        return face.mul(2.).sub(1.)  # scale [0,1]→[-1,1]

    def embed_tensor(self,face_tensor):
        """512-d L2-normalized embedding as a numpy array."""
        with torch.no_grad():
            emb = self.model(face_tensor.unsqueeze(0).to(DEVICE))
            emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()[0]

    # ── ORIGINAL TRAIN / TEST ───────────────────────────────────────────────────────
    def train_model(self, data_dir: str, celebrities: list, train_per_celeb: int = TRAIN_K):
        class_means = {}
        for celeb in celebrities:
            img_paths = sorted(glob.glob(os.path.join(data_dir, celeb, '*.*')))
            train_paths = img_paths[:train_per_celeb]
            embs = [self.embed_tensor(self.load_face_tensor(p)) for p in train_paths]
            mean_emb = np.mean(embs, axis=0)
            class_means[celeb] = mean_emb / np.linalg.norm(mean_emb)
        self.save_means(class_means)
        return class_means

    def test_model(self,model_name: str,
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
                emb =self.embed_tensor(self.load_face_tensor(p))
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

    def compute_gradient(self,class_means, image, celebrity):  # tensor [3,160,160], celebrity = 'scarlett', 'brad',

        # 1) build a stacked tensor of all class‐means: shape (512,5)
        cm_torch = torch.stack([torch.tensor(class_means[c], device=DEVICE, dtype=torch.float32)
                                for c in CELEBS], dim=1)
        # 2) prepare input for gradient
        x = image.unsqueeze(0).clone().detach().requires_grad_(True).to(DEVICE)  # (1,3,160,160)

        # 3) forward → embedding → normalize → compute 5 “logits”
        emb = self.model(x)                                          # (1,512)
        embn = F.normalize(emb, p=2, dim=1)                     # (1,512)
        logits = embn @ cm_torch                                # (1,5)

        y_true = CELEBS.index(celebrity)
        label = torch.tensor([y_true], dtype=torch.long)

        # 5) cross-entropy loss (we want to *increase* this)
        loss = F.cross_entropy(logits, label)
        loss.backward()

        # 6) FGSM step: move *with* the gradient sign to increase loss

        grad = x.grad   
        grad = grad.squeeze(dim = 0)                        # (1,3,160,160)
        

        return grad

    def forward(self, image): 
        class_means = self.load_vgg_means()
        emb = self.embed_tensor(image)
        emb = emb / np.linalg.norm(emb)
        sims = {c: np.dot(emb, class_means[c]) for c in CELEBS}
        pred = max(sims, key=sims.get)
        return pred

    def save_means(means): 
        np.save("./models/class_means.npy", means)

    def load_vgg_means(): 
        return np.load("./models/class_means.npy", allow_pickle=True).item()


    def compute_accuracy(self, test_set): 
        correct = 0
        total = 0
        for image, label in test_set: 
            celebrity = CELEBS[label]
            pred = self.forward(image)
            if(celebrity == pred): 
                correct += 1
            total += 1
        return (correct / total) * 100


# ── MAIN ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = VGGModel()
    model.train_model("Facenet", DATA_DIR, CELEBS, train_per_celeb = TRAIN_K)
