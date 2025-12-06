# NN-from-scratch

**Build Your Own Neural Network Library & Advanced Applications**
*MCT Program — CSE473s: Computational Intelligence (Fall 2025)*

---
## 👨‍🏫 Course Information

**Course:** CSE473s — Computational Intelligence
**Program:** MCT
**Semester:** Fall 2025
**Project:** Build Your Own Neural Network Library & Advanced Applications

---
## 👥 Team Members

| ID      | Name                      |
| ------- | ------------------------- |
| 2100543 | Mohamed Montasser         |
| 2100660 | Fatma Samy Ahmed          |
| 2101231 | Moaz Gamal Alsayed        |
| 2100961 | Mohamed Islam Salah Aldin |
| 2100820 | Zeyad Samer Lotfy         |

---
## 📌 Project Overview

This repository will host our implementation of a **neural network library from scratch** using only **Python and NumPy**.

We build:

- A minimal neural network library (layers, activations, loss, optimizer, and a Sequential model).  
- An MLP that learns the XOR problem.  
- An autoencoder for MNIST image reconstruction.  
- An SVM classifier trained on the autoencoder’s latent features.  
- The same architectures again using TensorFlow/Keras, to compare implementation effort, training time, and performance.

## Features

- **Core library (NumPy only)**  
  - `Layer` base class with `forward` and `backward`  
  - `Dense` fully connected layer  
  - Activations: ReLU, Sigmoid, Tanh, Softmax  
  - `MSE` loss  
  - `SGD` optimizer  
  - `Sequential` model to chain layers and run train steps  

- **XOR demo**  
  - Network: 2–4–1 MLP with Tanh + Sigmoid  
  - Trained with SGD + MSE to correctly classify all 4 XOR inputs.[2][3]

- **Autoencoder on MNIST**  
  - Encoder: 784 → 256 → 64 with ReLU  
  - Decoder: 64 → 256 → 784 with ReLU/Sigmoid  
  - Trained with MSE (input = target) for reconstruction.[1]

- **Latent-space SVM classifier**  
  - Use encoder to map MNIST images into 64‑dim latent vectors.  
  - Train an SVM on these features to classify digits and report accuracy, confusion matrix, and metrics.[4]

- **Gradient checking**  
  - Numerical gradient checking to verify backprop using  
    \(\partial L / \partial W \approx [L(W + \varepsilon) - L(W - \varepsilon)] / (2\varepsilon)\).[5][6]

- **TensorFlow/Keras baselines**  
  - Rebuild XOR MLP and autoencoder in TensorFlow/Keras.  
  - Compare training time and final reconstruction loss to the NumPy version.[1]

---

## Repository Structure

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── lib/
│   ├── __init__.py
│   ├── layers.py
│   ├── activations.py
│   ├── losses.py
│   ├── optimizer.py
│   └── network.py
└── notebooks/
    └── project_demo.ipynb
```

- `lib/` contains the neural network library (pure NumPy).  
- `notebooks/project_demo.ipynb` runs all demos: gradient checking, XOR, autoencoder, SVM on latent space, and TensorFlow/Keras comparison.


---

## Learning Goals

This project is meant to:

- Reinforce understanding of forward/backward propagation and gradient-based optimization.  
- Show how autoencoders can be used for dimensionality reduction and feature extraction.  
- Demonstrate transfer learning by training an SVM on learned latent features.  
- Illustrate the difference between a low-level NumPy implementation and a high-level deep learning framework (TensorFlow/Keras).
