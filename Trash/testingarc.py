import cv2
import numpy as np
from insightface.model_zoo import get_model
from numpy import dot
from numpy.linalg import norm
from data import TRAIN_SET

# Load the recognition model directly
model = get_model('buffalo_l', download=True)  # or manually load .onnx if you want
model.prepare(ctx_id=-1)

# Load pre-cropped 160x160 face image
img = cv2.imread("/Users/derekkirschbaum/cs190a/CS190AProject-s25/Dataset/Brad Pitt/000001.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img3 = cv2.imread("/Users/derekkirschbaum/cs190a/CS190AProject-s25/Dataset/Brad Pitt/000002.jpg")
img_rgb3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)


# Run inference
embedding1 = model.get_feat(img_rgb).flatten()
print("Embedding shape:", embedding1.shape)

embedding3 = model.get_feat(img_rgb3).flatten()
print("Embedding shape:", embedding3.shape)

img2 = cv2.imread("/Users/derekkirschbaum/cs190a/CS190AProject-s25/Dataset/Angelina Jolie/000001.jpg")
img_rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

embedding2 = model.get_feat(img_rgb2).flatten()
print("Embedding shape:", embedding2.shape)

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

sim = cosine_similarity(embedding1, embedding2)
print("Cosine similarity:", sim)

sim2 = cosine_similarity(embedding1, embedding3)
print("Cosine similarity:", sim2)

sim3 = cosine_similarity(embedding3, embedding2)
print("Cosine similarity:", sim3)


image_tensor, label = TRAIN_SET[0]

# Undo normalization (normalize was: (x - 0.5) / 0.5)
image_tensor_unnorm = image_tensor * 0.5 + 0.5  # [0,1]

# Convert to numpy image HWC RGB
image_np = image_tensor_unnorm.permute(1, 2, 0).cpu().numpy()
image_np = (image_np * 255).astype(np.uint8)

image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
cv2.imwrite('output_image.jpg', image_bgr)

# Now get embedding
embedding4 = model.get_feat(image_np).flatten()
print("Embedding shape:", embedding4.shape)

embeddings = []
labels = []

for i in range(20):
    image_tensor, label = TRAIN_SET[i]
    
    # Undo normalization (from (-1,1) back to (0,1))
    image_tensor_unnorm = image_tensor * 0.5 + 0.5
    
    # Convert tensor to numpy HWC RGB
    image_np = image_tensor_unnorm.permute(1, 2, 0).cpu().numpy()
    image_np_uint8 = (image_np * 255).astype(np.uint8)
    
    # Save the image as JPG (convert RGB->BGR)
    image_bgr = cv2.cvtColor(image_np_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'output_image_{i}.jpg', image_bgr)
    
    # Get embedding from ArcFace model
    embedding = model.get_feat(image_np_uint8).flatten()
    embeddings.append(embedding)
    labels.append(label)

# Now compute cosine similarity confusion matrix
num = len(embeddings)
confusion_matrix = np.zeros((num, num))

for i in range(num):
    for j in range(num):
        confusion_matrix[i, j] = cosine_similarity(embeddings[i], embeddings[j])

np.set_printoptions(precision=3, suppress=True)
print("Labels:", labels)
print("Cosine Similarity Confusion Matrix:\n", confusion_matrix)