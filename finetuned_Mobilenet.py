import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import joblib
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

embedding_model_2 = tf.keras.models.load_model("m2/fine_tuned_embedding.h5")
pca_2_loaded = joblib.load("m2/pca_2.joblib")
nn_index_2_loaded = joblib.load("m2/nn_index_2.joblib")

train_emb_2 = np.load("m2/train_emb_2.npy")
train_labels_2 = np.load("m2/train_labels_2.npy")
train_paths_2 = np.load("m2/train_paths_2.npy", allow_pickle=True)

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


def get_finetuned_embedding(img, embedding_model):
    img_255 = img * 255.0
    img_processed = preprocess_input(img_255)
    emb = embedding_model.predict(np.expand_dims(img_processed, axis=0))
    return emb.flatten()

def retrieve_similar_images_finetuned(query_img_path, embedding_model, pca, nn_index, k=5):
    img = load_and_preprocess_image(query_img_path)  # same utility as before
    if img is None:
        return []
    emb = get_finetuned_embedding(img, embedding_model)
    emb_pca = pca.transform([emb])
    distances, indices = nn_index.kneighbors(emb_pca, n_neighbors=k)
    return indices[0], distances[0]

# Example usage
query_path = 'input path here'
inds, dists = retrieve_similar_images_finetuned(query_path,
                                                embedding_model_2,
                                                pca_2_loaded,
                                                nn_index_2_loaded,
                                                k=5)

print("Retrieved indices:", inds)
print("Distances:", dists)
for idx in inds:
    print("Label:", train_labels_2[idx], " - Path:", train_paths_2[idx])


def plot_images_with_labels(input_img_path, retrieved_indices, labels, paths, k=5):
    """
    I used this function to plot the input image along with the k retrieved images and their labels.
    """
    input_img = load_and_preprocess_image(input_img_path)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns
    axes[0, 0].imshow(input_img)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis('off')

    for i in range(k):
        row, col = divmod(i + 1, 3)  
        retrieved_img = load_and_preprocess_image(paths[retrieved_indices[i]])  
        axes[row, col].imshow(retrieved_img)
        axes[row, col].set_title(f"Label: {labels[retrieved_indices[i]]}")
        axes[row, col].axis('off')

    if k < 5:
        for i in range(k + 1, 6):
            row, col = divmod(i, 3)
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()

plot_images_with_labels(
    query_path,
    inds,
    train_labels_2,
    train_paths_2,
    k=5
)
