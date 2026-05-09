# Gender Classification: A Comparative Deep Learning Study

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## Overview
This project implements an end-to-end computer vision pipeline to classify human faces as Male or Female. Rather than just building a single model, this repository serves as a comparative study between building a **Custom Convolutional Neural Network (CNN) from scratch** and utilizing **Transfer Learning via ResNet18**. 

The goal is to demonstrate fundamental deep learning architecture design, data pipeline automation, and industry-standard fine-tuning techniques.

---

<!-- 
NOTE TO SELF: UNCOMMENT THIS SECTION ONCE THE FINAL GRAPHS ARE PUSHED

### Training Loss vs. Validation Loss
<div align="center">
  <img src="loss_curve.png" alt="CNN Loss" width="45%">
  <img src="resnet_loss_curve.png" alt="ResNet Loss" width="45%">
</div>

### Confusion Matrices
<div align="center">
  <img src="confusion_matrix.png" alt="CNN Confusion Matrix" width="45%">
  <img src="resnet_confusion_matrix.png" alt="ResNet Confusion Matrix" width="45%">
</div>
-->

---

## Future Enhancements (In Progress)
To push the model accuracy closer to 99% and ensure robust generalization, the following optimizations are currently being implemented:
* **Data Augmentation:** Introducing random horizontal flips, rotations, and color jitter to the PyTorch `DataLoader` to artificially expand the training variance.
* **Learning Rate Scheduling:** Implementing `torch.optim.lr_scheduler` to decay the learning rate dynamically as the validation loss plateaus.
* **Full Network Fine-Tuning:** Unfreezing the early feature-extraction layers of the ResNet18 model to allow domain-specific adaptation to facial features.
## Model Architectures

### 1. Custom CNN (Baseline)
A lightweight convolutional neural network built to establish a performance baseline and demonstrate architectural fundamentals.
* **Structure:** 3 Convolutional Blocks (Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d)
* **Regularization:** Batch Normalization to stabilize gradients and 50% Dropout in the fully connected layer to prevent overfitting.
* **Loss & Optimizer:** Binary Cross Entropy (via CrossEntropyLoss logits) and Adam Optimizer.

### 2. ResNet18 (Transfer Learning)
An industry-standard architecture leveraging weights pre-trained on ImageNet.
* **Methodology:** The core feature extraction layers were frozen (`requires_grad = False`). Only the final fully connected classification head was replaced and trained specifically on the facial dataset, resulting in rapid convergence.

---

## Project Structure

```text
Gender-Classification/
├── data/
│   ├── raw/                 # Downloaded Kaggle dataset
│   └── test_images/         # Unseen images for inference testing
├── src/
│   ├── data_loader.py       # PyTorch Dataset & DataLoader pipeline
│   ├── model.py             # Custom CNN architecture
│   ├── transfer_model.py    # ResNet18 modification script
│   ├── train.py             # Training loop with checkpointing
│   ├── train_resnet.py      # ResNet-specific training loop
│   ├── evaluate.py          # Metric generation (Precision/Recall/F1)
│   └── predict.py           # Single-image inference script
├── .gitignore
├── requirements.txt
└── README.md