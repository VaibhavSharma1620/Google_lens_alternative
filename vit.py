
import cv2
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import time
import joblib
import numpy as np
IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_loaded = torch.load("m4_vit/vit_finetuned_full.pth", map_location=device)
model_loaded.eval()

pca_loaded = joblib.load("m4_vit/vit_pca.joblib")
nn_loaded  = joblib.load("m4_vit/vit_nn.joblib")

train_emb_pca_loaded = np.load("m4_vit/train_emb_pca.npy")
train_labels_loaded   = np.load("m4_vit/train_labels.npy")
train_paths_loaded    = np.load("m4_vit/train_paths.npy", allow_pickle=True)

inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def extract_vit_features(model, x):
    model.eval()
    with torch.no_grad():
        feats = model.forward_features(x)
    return feats


def retrieve_similar_vit(query_path, model, pca, nn_index, k=5):
    img_bgr = cv2.imread(query_path)
    if img_bgr is None:
        print("Could not load:", query_path)
        return [], []
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    pil_tensor = inference_transform(img_pil).unsqueeze(0).to(device)
    feats = extract_vit_features(model, pil_tensor).cpu().numpy().flatten().reshape(1, -1)

    # PCA
    feats_pca = pca.transform(feats)
    
    # NN search
    dists, inds = nn_index.kneighbors(feats_pca, n_neighbors=k)
    return inds[0], dists[0]

    

def plot_retrieved_images(indices, distances, train_paths, k=5):
    """Plot the top-k retrieved images in a single row."""
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices[:k]):
        path = train_paths[idx]
        
        # Load the image from disk
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, k, i+1)
        plt.imshow(img_rgb)
        plt.title(f"dist={distances[i]:.2f}\nidx={idx}")
        plt.axis("off")
    plt.show()

start=time.time()
query_path = 'input path here'
inds, dists = retrieve_similar_vit(query_path, model_loaded, pca_loaded, nn_loaded, k=5)
print("Indices:", inds)
print("Distances:", dists)

plot_retrieved_images(inds, dists, train_paths_loaded, k=5)
end=time.time()
