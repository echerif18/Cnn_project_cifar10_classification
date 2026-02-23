# 🧠 CIFAR-10 Image Classification — Deep Learning Project

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Transfer%20Learning-MobileNetV2-orange" alt="Transfer Learning">
  <img src="https://img.shields.io/badge/Experiment%20Tracking-W%26B-FFBE00?logo=weightsandbiases&logoColor=black" alt="Weights & Biases">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
</p>

> End-to-end CIFAR-10 image classification using custom CNNs and transfer learning, with full experiment tracking via **Weights & Biases (W&B)**.  

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Repository Structure](#-repository-structure)
- [Notebooks](#-notebooks)
- [Experiment Tracking (W&B)](#-experiment-tracking--wb)
- [How to Reproduce](#-how-to-reproduce)
- [Technology Stack](#-technology-stack)
- [Authors](#-authors)
- [License](#-license)

---

## 🔍 Project Overview

This project explores multiple deep learning approaches to classify images from the CIFAR-10 dataset into 10 categories:

> ✈️ Airplane · 🚗 Automobile · 🐦 Bird · 🐱 Cat · 🦌 Deer · 🐶 Dog · 🐸 Frog · 🐴 Horse · 🚢 Ship · 🚛 Truck

The project is structured in two main notebooks:

1. **Baseline CNN** — build, train, and iteratively improve a custom CNN from scratch
2. **Transfer Learning** — leverage pretrained architectures (MobileNetV2) on CIFAR-10

Key goals:
- Understand the impact of regularization, data augmentation, and early stopping on model performance
- Compare custom CNN architectures vs. pretrained transfer learning models
- Track, compare, and reproduce all experiments using **Weights & Biases**
- Save and share trained model artifacts via W&B

---

## 📦 Dataset

**CIFAR-10** — 60,000 color images (32×32 px) across 10 balanced classes:
- 50,000 training images / 10,000 test images

Source: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

The dataset is loaded directly via `tensorflow.keras.datasets.cifar10` — no manual download needed.

---

## 🗂 Repository Structure

```
cifar10_project/
├── project_deep_learning_baseline.ipynb       # CNN from scratch + improvements
├── project_deep_learning_transferLearning.ipynb  # Transfer learning with MobileNetV2
└── README.md
```

---

## 📓 Notebooks

### 1. `project_deep_learning_baseline.ipynb` — Custom CNN

This notebook walks through building a CNN from scratch with progressive improvements:

| Stage | Description |
|-------|-------------|
| **Baseline CNN** | Simple Conv → Pool → Dense architecture |
| **Regularization** | BatchNormalization + Dropout |
| **Deeper Model** | Multi-layer CNN (32 → 64 filters) |
| **Data Augmentation** | Rotation, flip, zoom, shift via `ImageDataGenerator` |
| **Early Stopping** | Patience-based stopping with best-weight restoration |
| **W&B Integration** | Full experiment logging, model artifact saving |

Key techniques used:
- Seeds fixed for reproducibility (`set_seeds(5)`)
- Optimizer: Adam with tunable learning rate
- Evaluation: Accuracy, Precision, Recall, F1-score, Confusion Matrix

---

### 2. `project_deep_learning_transferLearning.ipynb` — Transfer Learning

This notebook applies pretrained models to CIFAR-10 using two strategies:

| Model | Strategy | Image Size |
|-------|----------|------------|
| **MobileNetV2 (frozen)** | Feature extraction — base frozen, only head trained | 112×112 |
| **MobileNetV2 (unfrozen)** | Fine-tuning — last 20% of layers unfrozen | 112×112 |

Key techniques:
- Images upscaled to 112×112 (closest practical size to MobileNetV2's native 224×224)
- Preprocessing with `preprocess_input` specific to MobileNetV2
- Data augmentation applied during training
- Full W&B experiment tracking and model artifact logging/loading

---

## 📊 Experiment Tracking — W&B

All experiments are tracked using **[Weights & Biases](https://wandb.ai)**.

🔗 **Project dashboard:** [https://wandb.ai/Ironhack_cnn_project/cifar10_dataset_FantasticFour](https://wandb.ai/Ironhack_cnn_project/cifar10_dataset_FantasticFour)

Tracked for each run:
- Training & validation accuracy / loss per epoch
- Final evaluation metrics (train vs. test)
- Hyperparameters (learning rate, batch size, epochs, architecture)
- Model artifacts (`.keras` files stored and versioned in W&B)

To use W&B, authenticate once:
```bash
wandb login
```

Models are logged as W&B Artifacts and can be pulled by any teammate:
```python
import wandb
wandb.init(entity="Ironhack_cnn_project", project="cifar10_dataset_FantasticFour")
artifact = wandb.use_artifact("Ironhack_cnn_project/cifar10_dataset_FantasticFour/<model_name>:latest")
artifact.download()
```

---

## 🚀 How to Reproduce

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd cifar10_project
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install tensorflow wandb numpy pandas matplotlib seaborn scikit-learn
```

### 4. Authenticate with W&B

```bash
wandb login
```
---

## 🛠 Technology Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| Deep Learning | TensorFlow / Keras |
| Pretrained Models | MobileNetV2 (ImageNet weights) |
| Experiment Tracking | Weights & Biases (W&B) |
| Data & Utilities | NumPy, pandas |
| Visualization | Matplotlib, Seaborn |
| Evaluation | scikit-learn (classification report, confusion matrix) |
| Environment | Jupyter Notebook |

---

## 📜 License & Acknowledgments

This project is licensed under the **MIT License**.

This work was built together with [@alexcardenasgutierrez-droid](https://github.com/alexcardenasgutierrez-droid), [@aitor-vergaramartin](https://github.com/aitor-vergaramartin) and [@suzanacracco-max](https://github.com/suzanacracco-max) 🙌