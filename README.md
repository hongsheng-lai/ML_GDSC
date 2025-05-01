# ML_GDSC

This repository contains machine learning workflows applied to the GDSC (Genomics of Drug Sensitivity in Cancer) dataset, including data preprocessing, feature engineering, regression modeling, and neural network training.

## Contents

### 📊 Data Analysis
- **`data_analysis.ipynb`**  
  Performs PCA (Principal Component Analysis) and clustering to explore the structure and patterns in the GDSC dataset.

- **`data.ipynb`**  
  Handles data cleaning and feature engineering tasks to prepare the GDSC dataset for model training.

### 🤖 Model Training

- **`main.py`**, **`utils/`**, **`results/`**, **`config.yaml`**  
  Implements and evaluates a Multilayer Perceptron (MLP) model for predicting outcomes based on the GDSC features.

- **`regression.ipynb`**  
  Trains and evaluates classical regression models, comparing performance across methods.

- **`scratch_regression.py`**  
  Implements regression models (Linear Regression, Decision Tree, Random Forest, Gradient Boosting) from scratch. Final results are shown to match those from `sklearn`.
