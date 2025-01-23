import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import joblib
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
class_names =['Backpacks',
 'Belts',
 'Bra',
 'Briefs',
 'Casual Shoes',
 'Deodorant',
 'Dresses',
 'Earrings',
 'Flats',
 'Flip Flops',
 'Formal Shoes',
 'Handbags',
 'Heels',
 'Jeans',
 'Kurtas',
 'Nail Polish',
 'Perfume and Body Mist',
 'Sandals',
 'Sarees',
 'Shirts',
 'Shorts',
 'Socks',
 'Sports Shoes',
 'Sunglasses',
 'Tops',
 'Trousers',
 'Tshirts',
 'Wallets',
 'Watches']
IMG_SIZE   = 224
BATCH_SIZE = 32 

inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
class EmbeddingNet(nn.Module):
    """i didnt use mobilenet here but a simpler smaller CNN model which give 128 size embedding at the end"""
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(128*(IMG_SIZE//8)*(IMG_SIZE//8), embedding_dim)
        
    def forward(self, x):
        # x: (B,3,224,224)
        x = F.relu(self.conv1(x))  # -> (B,32,224,224)
        x = self.pool(x)           # -> (B,32,112,112)
        
        x = F.relu(self.conv2(x))  # -> (B,64,112,112)
        x = self.pool(x)           # -> (B,64,56,56)
        
        x = F.relu(self.conv3(x))  # -> (B,128,56,56)
        x = self.pool(x)           # -> (B,128,28,28)
        
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)             # (B, embedding_dim)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = EmbeddingNet(embedding_dim=128).to(device)
loaded_model.load_state_dict(torch.load("m5_siamese/triplet_base_model.pth", map_location=device))
loaded_model.eval()

pca_loaded     = joblib.load("m5_siamese/triplet_pca.joblib")
nn_loaded      = joblib.load("m5_siamese/triplet_nn.joblib")

train_emb_pca_loaded = np.load("m5_siamese/train_emb_pca.npy")
train_labels_loaded  = np.load("m5_siamese/train_labels.npy")
train_paths_loaded   = np.load("m5_siamese/train_paths.npy", allow_pickle=True)




def get_triplet_embedding(model, img_tensor):
    model.eval()
    with torch.no_grad():
        emb = model(img_tensor)
    return emb.cpu().numpy().flatten()

def retrieve_similar_triplet(query_path, model, pca, nn_index, k=5):
    # load via cv2
    img_bgr = cv2.imread(query_path)
    if img_bgr is None:
        print("Could not load image:", query_path)
        return [], []
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # (H,W,3) => (3,H,W)
    tensor = torch.from_numpy(img_rgb).permute(2,0,1).float()/255.0
    tensor = inference_transform.transforms[0](tensor)  
    tensor = transforms.functional.resize(tensor, (IMG_SIZE, IMG_SIZE))
    tensor = tensor.unsqueeze(0).to(device)
    
    
    emb = get_triplet_embedding(model, tensor)  # shape (128,)
    emb_pca = pca.transform([emb])             # shape (1, pca_dim)
    
    dists, inds = nn_index.kneighbors(emb_pca, n_neighbors=k)
    return inds[0], dists[0]
def plot_retrieved_images(indices, distances, train_paths, k=5):
    plt.figure(figsize=(15,3))
    for i, idx in enumerate(indices[:k]):
        path = train_paths[idx]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, k, i+1)
        plt.imshow(img_rgb)
        plt.title(f"dist={distances[i]:.2f}\nidx={idx}")
        plt.axis("off")
    plt.show()


query_path = "input path here"
inds, dists = retrieve_similar_triplet(query_path, loaded_model, pca_loaded, nn_loaded, k=5)

print("Retrieved indices:", inds)
print("Distances:", dists)
print("Retrieved labels:", train_labels_loaded[inds])
for i in inds:
    print("Class Name:", class_names[ train_labels_loaded[i] ], "Path:", train_paths_loaded[i])

# Plot them
plot_retrieved_images(inds, dists, train_paths_loaded, k=5)

