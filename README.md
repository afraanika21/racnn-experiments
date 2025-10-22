

# Retrieval-Augmented Robustness — Experiments with RaCNN, ViT & Mixup Variants

This repository explores **Retrieval-Augmented Convolutional Networks (RaCNN)** and their modern variants to study adversarial robustness and manifold regularization techniques.

---

## 📍Project Overview

| Experiment | Description |
|-------------|-------------|
| `ra-cnn-resnet` | Replication of Cho et al. (NeurIPS 2019) using ResNet-18 backbone and FAISS retrieval engine. |
| `ra-cnn-vit` | RaCNN modified with a Vision Transformer (ViT) backbone instead of CNN. |
| `ra-cnn-automixup` | Incorporates AutoMixup (AAAI 2021) as an alternative to the local mixup used in RaCNN. |

---

## 🧠 Background

The RaCNN architecture combines:
- A **retrieval engine (φ′)** to construct a local convex hull of neighbors,
- A **projection module** using attention weights to map inputs back onto the data manifold,
- And **local mixup regularization** to improve linearity within the manifold.

This approach addresses **both on-manifold and off-manifold adversarial attacks**.

---

## ⚙️ Setup

```bash
pip install torch torchvision faiss-cpu torchattacks tqdm matplotlib
````

---

## 📁 Folder Structure

```
notebooks/
  ra-cnn-resnet/     → paper replication (ResNet backbone)
  ra-cnn-vit/        → transformer backbone variant
  ra-cnn-automixup/  → alternative mixup function
src/                 → reusable modules (retrieval, backbones, projection, etc.)
docs/                → experiment logs & diagrams
```

---


---

## 📚 References

* Cho et al., *Retrieval-Augmented Convolutional Neural Networks against Adversarial Examples*, NeurIPS 2019.
* Dosovitskiy et al., *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*, ICLR 2021.
* Qin et al., *Adversarial AutoMixup*, ICLR 2024.


````