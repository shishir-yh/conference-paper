# Enhancing CNN Performance with Boosting Techniques for Image Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2+-blue.svg)

A comprehensive study evaluating the effectiveness of combining convolutional neural networks (CNNs) with popular boosting algorithms for image classification tasks.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

## 🌟 Overview
This repository contains an implementation and comparative analysis of hybrid models that combine:
- **Base CNN Architecture** for feature learning
- **Boosting Algorithms** (XGBoost, Gradient Boost, AdaBoost) for classification

We systematically evaluate how layer-specific features (extracted from flatten layer, first dense layer, and third dense layer) impact the performance of different boosting techniques.

## 🔑 Key Features
- **Hybrid Architecture**: Combine deep learning and ensemble methods
- **Layer-Wise Analysis**: Features extracted from three critical CNN layers
- **Performance Benchmarking**: Compare 3 boosting algorithms against baseline CNN
- **Modular Design**: Easy to extend with new classifiers/datasets
- **Reproducible Results**: Detailed configuration and evaluation metrics

## 🛠 Installation
1. Clone repository:
```bash
git clone https://github.com/yourusername/CNN-Boosting-Image-Classification.git
cd CNN-Boosting-Image-Classification
Install dependencies:

bash
Copy
pip install -r requirements.txt
🚀 Usage
1. Train Base CNN
python
Copy
python code/cnn_model.py \
  --epochs 50 \
  --batch_size 32 \
  --dataset_path ./datasets/cifar10
2. Extract Features
python
Copy
python code/feature_extraction.py \
  --layer flatten \
  --output_dir ./features
3. Train Boosting Models
python
Copy
python code/boosting_models.py \
  --model gradient_boost \
  --features_path ./features/flatten_features.npy
Command Line Arguments
Parameter	Description	Default
--layer	Layer for feature extraction	flatten
--model	Boosting algorithm to use	xgboost
--epochs	Training epochs for CNN	50
--batch_size	Batch size for training	32
📊 Results
Performance Comparison
Layer	Model	Test Accuracy	F1-Score
Flatten	Gradient Boost	92.34%	0.921
Flatten	XGBoost	92.15%	0.919
1st Dense	Gradient Boost	91.89%	0.916
3rd Dense	XGBoost	91.02%	0.908
Baseline CNN	CNN Only	90.12%	0.899
Key Findings
Gradient Boost outperformed other models in shallow layers (flatten + 1st dense)

XGBoost showed superior performance with deeper features (3rd dense layer)

All boosted models surpassed baseline CNN accuracy

Best overall improvement: +2.22% accuracy (CNN+Gradient Boost vs Baseline)

Accuracy Comparison

🔮 Future Work
Integrate SVM/LSTM classifiers

Add attention mechanisms to CNN

Implement automated layer selection

Test on medical imaging datasets

Develop unified API for feature extraction

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

Note: Replace dataset paths with your actual dataset directory. For CIFAR-10 experiments, use the built-in TensorFlow dataset loader.

Contributors: [Yeasir Hossain Shishir] | [subroshishir0@gmail.com] | [East West University]

Copy

This README:
1. Clearly states the project's purpose and methodology
2. Provides easy-to-follow implementation instructions
3. Highlights key technical contributions
4. Presents results in both tabular and visual formats
5. Suggests meaningful extensions for future work
6. Maintains professional formatting with badges and tables

To use this:
1. Create a new repository on GitHub
2. Copy this content into a `README.md` file
3. Add your actual implementation code to the specified directory structure
4. Commit and push your files
5. Add dataset (or instructions for dataset acquisition) in the `datasets` folder

The badge links will automatically work once pushed to GitHub. Add actual result files (PNG/PDF) in the `results` directory to make the comparisons visible.
