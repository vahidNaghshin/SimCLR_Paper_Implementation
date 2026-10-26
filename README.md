
# SimCLR on CIFAR-10 (Step-by-Step PyTorch Implementation)

This repository contains a simple, educational, and modular step-by-step implementation of **SimCLR (A Simple Framework for Contrastive Learning of Visual Representations)** on the CIFAR-10 dataset using PyTorch.

## 🚀 Project Overview

SimCLR learns high-quality visual representations without human labels by maximizing agreement between differently augmented views of the same image (positive pairs) while minimizing agreement with other images (negative pairs) via a contrastive loss objective.

```
+------------------+      +---------------------+      +-----------------+
|  Input Image (x) | ---> | Augmented Views     | ---> | Base Encoder (f)|
|                  |      | (x_i, x_j)          |      | ResNet-18       |
+------------------+      +---------------------+      +-----------------+
                                                                |
                                                                v
                                                       +-----------------+
                                                       | Proj. Head (g)  |
                                                       | 2-layer MLP (z) |
                                                       +-----------------+
                                                                |
                                                                v
                                                       +-----------------+
                                                       |  NT-Xent Loss   |
                                                       +-----------------+
```

## 🛠️ Key Components implemented

1. **Stochastic Data Augmentation (`ContrastiveTransformations`)**
   - Applies random cropping, horizontal flipping, paper-scaled color jittering, and Gaussian blur to generate two correlated views per image.

2. **Base Encoder $f(\cdot)$ (`SimCLRBaseEncoder`)**
   - A customized **ResNet-18** adapted for $32 \times 32$ pixels by replacing the initial large kernel convolutional layer (`conv1`) and removing the early max-pooling layer to prevent rapid downsampling.

3. **Projection Head $g(\cdot)$ (`ProjectionHead`)**
   - A 2-layer Multilayer Perceptron (MLP) with a **ReLU** non-linearity to project features from 512-D space to a 128-D latent space where contrastive loss is applied.

4. **NT-Xent Loss (`NTXentLoss`)**
   - A vectorized normalized temperature-scaled cross-entropy loss function that computes pairwise cosine similarity across $2N$ views while masking self-similar elements.

## 📈 Training Progress

Using a batch size of 256 and training on a GPU, the model demonstrates a consistent reduction in contrastive loss:

* **Epoch 1:** Average NT-Xent Loss: `5.4393`
* **Epoch 5:** Average NT-Xent Loss: `5.0955`

## ⚙️ How to Use

Run the cells sequentially in the provided Jupyter Notebook. The codebase utilizes modern `torchvision` models and standard libraries (`torch`, `matplotlib`, `PIL`).

