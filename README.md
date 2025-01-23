# Google Lens Alternative: Image Similarity Search Comparison

## Section 1: Methods 

### 1. Pretrained CNN (MobileNet) + PCA + Nearest Neighbor
-  Leverages transfer learning with a lightweight, efficient model
-  Utlize Separable convolution property to reduce the number of paramter  

### 2. Fine-tuned CNN (MobileNet) + PCA + Nearest Neighbor
- Adapts pre-trained model to specific domain
- To get Better feature representation for target use case

### 3. Fine-tuned Vision Transformer (ViT) + PCA + Nearest Neighbor
- Leverages transformer architecture's powerful representation learning
- Adaptable to complex visual features with Global Attention

### 4. Fine-tuned Autoencoder + PCA + Nearest Neighbor
- Learns compact, information-dense representations
- reduce dimension and Captures essential image features

### 5. Siamese Network + Triplet Loss
- Direct learning of similarity metric

## Section 2: Performance Comparison

### Comparative Results

| Method | Precision@k |  Retrieval Time (s) | 
|--------|-----------|--------|
| Pretrained CNN + PCA | 0.967 | 0.57 |
| Fine-tuned CNN + PCA | 0.964 | 0.65 | 
| Fine-tuned ViT + PCA | 0.959 | 3.96 |
| Fine-tuned Autoencoder + PCA | 0.911 | 0.09 |
| Siamese Network + Triplet Loss | 0.89 | 0.54 |

## Section 3: Conclusion and future imporvements

- using FAISS instead of Nearest Neighbour
- Adjusting param like PCA dimension
- Can try Contrastive loss (CLIP model embeddings also)
- Add dropout or other regularization techniques for prone to overfit model like VIT
- Training on full dataset
- references:
  - https://arxiv.org/pdf/1503.03832
  - https://arxiv.org/pdf/1704.04861
