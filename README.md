# Deepfake Detection Using EfficientNet and GRU-Based Hybrid Model

![Python](https://img.shields.io/badge/Python-3.8-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Overview

The project proposes a hybrid deep learning architecture for detecting deepfake videos by combining:

- **EfficientNetB0** for lightweight and efficient spatial feature extraction
- **Gated Recurrent Units (GRU)** for modeling temporal dependencies between video frames

The model effectively identifies subtle spatial-temporal inconsistencies found in manipulated content.

---

## 🚀 Key Features

- Multi-scale spatial feature extraction using pre-trained **EfficientNetB0**
- Sequential learning via **GRU** to detect frame-level inconsistencies
- Lightweight design suitable for real-time and low-resource environments
- Achieved **89.71% accuracy** and **94.56% AUC-ROC** on benchmark datasets

---

## 📊 Results

| Metric        | Score       |
|---------------|-------------|
| Accuracy      | 89.71%      |
| Precision     | 88.18%      |
| Recall        | 82.11%      |
| F1-Score      | 85.04%      |
| AUC-ROC       | 94.56%      |

Tested on combined datasets: **Celeb-DF (v2)** and **DFDC**

---

## 🧠 Architecture

See [`model_architecture.png`](./model_architecture.png) for visual representation.

---

## 📁 Dataset

- **Celeb-DF (v2)**: [Celeb-DF](https://github.com/yuezunli/Celeb-DF)
- **DFDC**: [Deepfake Detection Challenge Dataset](https://ai.facebook.com/tools/datasets/dfdc)

*Note: Due to licensing restrictions, datasets are not included in this repo. Please download from official sources.*

---




