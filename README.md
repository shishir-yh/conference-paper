# Enhancing CNN Performance with Boosting Techniques for Image Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-blue.svg)

A comprehensive study evaluating the effectiveness of combining convolutional neural networks (CNNs) with popular boosting algorithms for image classification tasks.

---

## 📌 Table of Contents
- [🌟 Overview](#-overview)
- [🔑 Key Features](#-key-features)
- [🚀 Usage](#-usage)
- [📊 Results](#-results)
- [🔮 Future Work](#-future-work)
- [📜 Certificate of Publication](#-certificate-of-publication)
---

## 🌟 Overview
This repository contains an implementation and comparative analysis of hybrid models that combine:
- **Base CNN Architecture** for feature learning
- **Boosting Algorithms** (XGBoost, Gradient Boost, AdaBoost) for classification

We systematically evaluate how layer-specific features (extracted from the flatten layer, first dense layer, and third dense layer) impact the performance of different boosting techniques.

---

## 🔑 Key Features
✅ **Hybrid Architecture**: Combines deep learning and ensemble methods  
✅ **Layer-Wise Analysis**: Features extracted from three critical CNN layers  
✅ **Performance Benchmarking**: Compare three boosting algorithms against a baseline CNN  
✅ **Modular Design**: Easy to extend with new classifiers/datasets  
✅ **Reproducible Results**: Detailed configuration and evaluation metrics provided  

---

## 🚀 Usage

### 1️⃣ Train Base CNN
```bash
python code/cnn_model.py --epochs 50 --batch_size 32 --dataset_path ./datasets/cifar10
```

### 2️⃣ Extract Features
```bash
python code/feature_extraction.py --layer flatten --output_dir ./features
```

### 3️⃣ Train Boosting Models
```bash
python code/boosting_models.py --model gradient_boost --features_path ./features/flatten_features.npy
```

### Command Line Arguments
| Parameter      | Description                         | Default |
|---------------|------------------------------------- |---------|
| `--layer`     | Layer for feature extraction         | flatten |
| `--model`     | Boosting algorithm to use            | xgboost |
| `--epochs`    | Training epochs for CNN              | 50      |
| `--batch_size`| Batch size for training              | 32      |

---

## 📊 Results

### Performance Comparison
| Layer       | Model            | Test Accuracy | F1-Score |
|------------|-----------------|--------------|----------|
| Flatten    | Gradient Boost  | 92.34%       | 0.921    |
| Flatten    | XGBoost         | 92.15%       | 0.919    |
| 1st Dense  | Gradient Boost  | 91.89%       | 0.916    |
| 3rd Dense  | XGBoost         | 91.02%       | 0.908    |
| Baseline CNN | CNN Only      | 90.12%       | 0.899    |

### 🔍 Key Findings
- **Gradient Boost outperformed other models** in shallow layers (flatten + 1st dense layer).
- **XGBoost showed superior performance** with deeper features (3rd dense layer).
- **All boosted models surpassed baseline CNN accuracy.**
- **Best overall improvement**: +2.22% accuracy (CNN + Gradient Boost vs Baseline).

---

## 🔮 Future Work
🔹 Integrate SVM/LSTM classifiers  
🔹 Add attention mechanisms to CNN  
🔹 Implement automated layer selection  
🔹 Test on medical imaging datasets  
🔹 Develop a unified API for feature extraction  

---
📜 Certificate of Publication

This research has been officially published. You can view the certificate of publication here:


---


## 👥 Contributors
[Yeasir Hossain Shishir] | [subroshishir0@gmail.com] | [East West University]

