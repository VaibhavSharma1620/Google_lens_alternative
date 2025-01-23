# Load
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import joblib
encoder_3 = load_model("m3_autoencoder/encoder.h5")  # we only need the encoder for retrieval
pca_3_loaded = joblib.load("m3_autoencoder/pca_3.joblib")
nn_index_3_loaded = joblib.load("m3_autoencoder/nn_index_3.joblib")
train_paths_ae__loaded    = np.load("m3_autoencoder/train_paths_ae.npy", allow_pickle=True)
train_ae_emb_pca_loaded = np.load("m3_autoencoder/train_ae_emb.npy")
train_ae_labels_loaded = np.load("m3_autoencoder/train_ae_labels.npy")

IMG_SIZE = 224

def load_and_preprocess_image(img_path, target_size=IMG_SIZE):
    """Load an image from disk, and preprocess it for example: resize and rescale"""
    img = cv2.imread(img_path)
    if img is None:
        return None  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size))
    img = img.astype('float32') / 255.0
    return img


# Example usage
query_path = 'input path here'


def retrieve_similar_images_ae(query_img_path, encoder_model, pca, nn_index, k=5):
    img = load_and_preprocess_image(query_img_path)
    if img is None:
        print("Could not load query image.")
        return []
    
    # Convert to embedding
    emb = encoder_model.predict(np.expand_dims(img, axis=0))
    emb_flat = emb.flatten()

    # Apply PCA
    emb_pca = pca.transform([emb_flat])

    # Retrieve neighbors
    distances, indices = nn_index.kneighbors(emb_pca, n_neighbors=k)
    return indices[0], distances[0]
def plot_query_and_retrieved(query_img_path, inds, dists, train_paths, train_labels, k=5):
    fig = plt.figure(figsize=(18, 3))
    
    # Plot query on the left
    query_img = load_and_preprocess_image(query_img_path)
    query_img_plot = (query_img * 255).astype("uint8")
    plt.subplot(1, k+1, 1)
    plt.imshow(query_img_plot)
    plt.title("Query Image")
    plt.axis('off')
    
    # Plot retrieved
    for i, idx in enumerate(inds[:k]):
        img_path = train_paths[idx]
        retrieved_label = train_labels[idx]
        retrieved_img = load_and_preprocess_image(img_path)
        retrieved_img_plot = (retrieved_img * 255).astype("uint8")
        
        plt.subplot(1, k+1, i+2)
        plt.imshow(retrieved_img_plot)
        plt.title(f"Label: {retrieved_label}\nDist: {dists[i]:.2f}")
        plt.axis('off')
    
    plt.show()
inds_3, dists_3 = retrieve_similar_images_ae(query_path, encoder_3, pca_3_loaded, nn_index_3_loaded)
plot_query_and_retrieved(query_path, inds_3, dists_3, train_paths_ae__loaded, train_ae_labels_loaded, k=5)