# Plant Disease Classification using Convolutional Neural Networks

Plant diseases are one of the biggest reasons for crop yield loss worldwide. Many infections start as small visual symptoms on leaves—spots, discoloration, mold-like patches, rust, or blight patterns. If farmers or growers detect these early, they can take action (remove infected plants, apply targeted treatment, isolate affected areas) before the disease spreads across the field.

In real life, disease identification is often done manually by farmers or agriculture experts. That process is slow, depends heavily on experience, and becomes difficult when symptoms look similar across different diseases or crops. In large farms, checking every plant regularly is not practical, which increases the chance of late detection.

This project builds a deep learning–based image classification system that can automatically recognize plant leaf diseases from photos. Using the PlantVillage dataset and a custom CNN model, the pipeline learns visual patterns of healthy vs infected leaves and predicts the most likely disease category. The goal is to demonstrate how computer vision can support faster, consistent, and scalable plant health monitoring.

![image alt](https://github.com/SimarSaka/LeafGuard-DL-Powered-Plant-Disease-Detection/blob/main/Mind_Map.png?raw=true)


---

## 1. Project Overview

The system is designed as a modular CNN-based classifier that processes plant leaf images and predicts the associated disease class.

- Framework: TensorFlow / Keras  
- Dataset: PlantVillage (Color images)  
- Architecture: Custom Sequential CNN  
- Input Resolution: 224 × 224 RGB  

 ![image alt](https://github.com/SimarSaka/LeafGuard-DL-Powered-Plant-Disease-Detection/blob/main/Fig01.jpeg?raw=true)

---

## 2. Reproducibility and Environment Setup

To ensure consistent results across multiple runs, deterministic behavior is enforced by fixing random seeds for Python, NumPy, and TensorFlow. This prevents run-to-run variation in weight initialization and data shuffling, making experiments reliable and debuggable.

 ![image alt](https://github.com/SimarSaka/LeafGuard-DL-Powered-Plant-Disease-Detection/blob/main/Fig02.jpeg?raw=true)

---

## 3. Automated Data Ingestion Strategy

The dataset is downloaded programmatically using KaggleHub. The pipeline dynamically locates the dataset root and automatically identifies the correct image directory, eliminating the need for manual dataset setup.

 ![image alt](https://github.com/SimarSaka/LeafGuard-DL-Powered-Plant-Disease-Detection/blob/main/Fig03.jpeg?raw=true)

---

## 4. Preprocessing and Data Generator Configuration

Image preprocessing is handled using `ImageDataGenerator` for efficient batch loading.

- Pixel rescaling to [0, 1]  
- Automatic 80/20 training–validation split  
- Target size: 224 × 224  
- Batch size: 32  

This setup supports stable training and memory efficiency.

 ![image alt](https://github.com/SimarSaka/LeafGuard-DL-Powered-Plant-Disease-Detection/blob/main/Fig04.jpeg?raw=true)

---

## 5. CNN Architecture: Custom Sequential Model

A lightweight CNN architecture is used to balance performance and computational cost.

Architecture flow:
- Convolution layers for feature extraction  
- MaxPooling layers for spatial reduction  
- Dense layers for classification  
- Softmax output for multi-class prediction  

 ![image alt](https://github.com/SimarSaka/LeafGuard-DL-Powered-Plant-Disease-Detection/blob/main/Fig05.jpeg?raw=true)

---

## 6. Feature Extraction Dynamics

As data passes through the network, feature depth increases while spatial resolution decreases. ReLU activation introduces non-linearity, and pooling layers reduce computation while preserving important features.

 ![image alt](https://github.com/SimarSaka/LeafGuard-DL-Powered-Plant-Disease-Detection/blob/main/Fig06.jpeg?raw=true)

---

## 7. Classification Head and Optimization

After feature extraction:
- Feature maps are flattened into a 1D vector  
- A dense layer performs high-level reasoning  
- Softmax outputs class probabilities  

Optimization details:
- Optimizer: Adam  
- Loss: Categorical Crossentropy  
- Metric: Accuracy  

 ![image alt](https://github.com/SimarSaka/LeafGuard-DL-Powered-Plant-Disease-Detection/blob/main/Fig07.jpeg?raw=true)

---

## 8. Training Loop Execution

The model is trained using a standard supervised learning loop with batch-wise ingestion. Validation performance is evaluated at the end of each epoch. A small number of epochs is used for rapid prototyping.

 ![image alt](https://github.com/SimarSaka/LeafGuard-DL-Powered-Plant-Disease-Detection/blob/main/Fig08.jpeg?raw=true)

---

## 9. Monitoring Convergence: Accuracy and Loss

Training progress is visualized using accuracy and loss curves.

Observations:
- Training and validation accuracy increase steadily  
- Training and validation loss decrease smoothly  
- Minimal gap between curves indicates limited overfitting  

 ![image alt](https://github.com/SimarSaka/LeafGuard-DL-Powered-Plant-Disease-Detection/blob/main/Fig09.jpeg?raw=true)

---

## 10. Quantitative Evaluation: Classification Report

For multi-class problems, accuracy alone is insufficient. Precision, recall, and F1-score are computed using Scikit-learn to provide a detailed, class-wise performance breakdown.

 ![image alt](https://github.com/SimarSaka/LeafGuard-DL-Powered-Plant-Disease-Detection/blob/main/Fig10.jpeg?raw=true)

---

## 11. Error Analysis using Confusion Matrix

The confusion matrix compares true labels with predicted labels. Strong diagonal values indicate correct predictions, while off-diagonal entries reveal specific class confusions that can guide further model improvements.

 ![image alt](https://github.com/SimarSaka/LeafGuard-DL-Powered-Plant-Disease-Detection/blob/main/Fig11.jpeg?raw=true)

---

## 12. Qualitative Validation with Sample Predictions

Sample predictions on unseen validation images are visualized to compare predicted labels with ground truth. This provides a human-readable sanity check and helps interpret model behavior beyond numerical metrics.

 ![image alt](https://github.com/SimarSaka/LeafGuard-DL-Powered-Plant-Disease-Detection/blob/main/Fig12.jpeg?raw=true)

---
