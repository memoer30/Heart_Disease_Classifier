# Heart Disease Classifier

A comprehensive machine learning project that predicts heart disease using various classification algorithms and hyperparameter optimization techniques.

## Overview

This project provides multiple approaches to heart disease prediction, from exploratory data analysis to optimized machine learning models. The project includes three Jupyter notebooks:

1. **end_to_end_heart_disease_classification.ipynb**: Complete end-to-end data science workflow including EDA, feature engineering, and model comparison (Logistic Regression, KNN, Random Forest)
2. **Heart_Disease_Classifier.ipynb**: Focused Random Forest implementation with hyperparameter optimization
3. **Heart_Disease_Classifier_Pipeline.ipynb**: Production-ready scikit-learn Pipeline implementation with hyperparameter tuning

Each notebook explores different aspects:
- **Baseline Models**: Multiple classifiers with default parameters
- **RandomizedSearchCV**: Efficient exploration of hyperparameter space
- **GridSearchCV**: Fine-tuned search for optimal parameters
- **Comprehensive EDA**: Visualization and correlation analysis

## Dataset

The project uses `heart-disease.csv` from the Cleveland database (UCI Machine Learning Repository), containing 303 patient records with 14 clinical features and a binary target variable indicating heart disease presence.

### Features
- **age**: Age in years
- **sex**: (1 = male; 0 = female)
- **cp**: Chest pain type (0-3)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
- **restecg**: Resting electrocardiographic results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes; 0 = no)
- **oldpeak**: ST depression induced by exercise
- **slope**: Slope of peak exercise ST segment (0-2)
- **ca**: Number of major vessels colored by fluoroscopy (0-3)
- **thal**: Thalium stress test result
- **target**: Heart disease diagnosis (1 = disease, 0 = no disease)

## Features

### End-to-End Workflow (`end_to_end_heart_disease_classification.ipynb`)
- **Comprehensive EDA** with visualizations:
  - Target distribution analysis
  - Heart disease frequency by sex and chest pain type
  - Age vs. maximum heart rate scatter plots
  - Correlation heatmap
- **Multiple algorithms**: Logistic Regression, K-Nearest Neighbors, Random Forest
- **Model comparison** across different classifiers
- **Cross-validation** for robust evaluation
- **ROC curves** and confusion matrices

### Standard Implementation (`Heart_Disease_Classifier.ipynb`)
- **Data preprocessing** and train/test split (80/20 ratio)
- **Baseline Random Forest** with default parameters
- **Hyperparameter optimization** using RandomizedSearchCV and GridSearchCV
- **Performance metrics**: Accuracy, Precision, Recall, F1-score
- **Visual comparison** of model performance

### Pipeline Implementation (`Heart_Disease_Classifier_Pipeline.ipynb`)
- **Production-ready scikit-learn Pipelines**
- **Clean code structure** preventing data leakage
- **Reproducible workflow** with proper cross-validation
- **Hyperparameter tuning** integrated into pipeline

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## Installation

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

### Option 1: End-to-End Analysis
1. Open `end_to_end_heart_disease_classification.ipynb`
2. Run all cells to perform:
   - Complete exploratory data analysis
   - Feature correlation analysis
   - Training and evaluation of Logistic Regression, KNN, and Random Forest
   - Hyperparameter tuning with RandomizedSearchCV and GridSearchCV
   - Comprehensive model comparison

### Option 2: Standard Random Forest Approach
1. Open `Heart_Disease_Classifier.ipynb`
2. Run all cells to:
   - Train baseline Random Forest model
   - Perform hyperparameter optimization
   - Compare baseline vs. optimized models

### Option 3: Pipeline Implementation
1. Open `Heart_Disease_Classifier_Pipeline.ipynb`
2. Run all cells for a production-ready pipeline approach with:
   - Clean, maintainable code structure
   - Proper cross-validation
   - Hyperparameter tuning integrated into pipeline

## Model Details

### Algorithms Tested
- **Logistic Regression**: Linear classification baseline
- **K-Nearest Neighbors (KNN)**: Instance-based learning
- **Random Forest**: Ensemble method with decision trees

### Baseline Random Forest
- Uses default scikit-learn parameters
- Establishes performance benchmark

### RandomizedSearchCV
Explores the following hyperparameter ranges:
- `n_estimators`: [100, 200, 500, 1000]
- `max_depth`: [None, 5, 10, 20, 30]
- `max_features`: ['sqrt', 'log2']
- `min_samples_split`: [2, 4, 6]
- `min_samples_leaf`: [1, 2, 4]
- `class_weight`: [None, 'balanced']

### GridSearchCV
Fine-tunes parameters based on RandomizedSearchCV results:
- `n_estimators`: [100, 200, 500]
- `max_depth`: [None]
- `max_features`: ['sqrt', 'log2']
- `min_samples_split`: [6]
- `min_samples_leaf`: [1, 2]

## Evaluation

The project uses comprehensive evaluation metrics:
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positive rate (positive predictive value)
- **Recall**: Sensitivity (true positive rate)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC Curves**: Visual representation of model performance
- **Confusion Matrix**: Detailed classification results
- **Cross-validation**: 5-fold CV for robust performance estimation

Custom `evaluate_preds()` function computes and displays all metrics. Results are compared visually using bar charts across different models and approaches.

## Results

The notebooks demonstrate:
- **Baseline performance** for multiple algorithms
- **Performance improvements** through hyperparameter tuning
- **Visual comparisons** of all tested models
- **Best practices** for model selection and optimization

Target: 95% accuracy for production deployment consideration

## Project Structure

```
Heart_Disease_Classifier/
├── end_to_end_heart_disease_classification.ipynb  # Complete ML workflow with EDA
├── Heart_Disease_Classifier.ipynb                 # Standard Random Forest implementation
├── Heart_Disease_Classifier_Pipeline.ipynb        # Production-ready Pipeline version
└── README.md                                      # This file
```

## Notebook Descriptions

### `end_to_end_heart_disease_classification.ipynb`
The most comprehensive notebook featuring:
- Problem definition and data dictionary
- Extensive exploratory data analysis
- Multiple visualizations (bar charts, scatter plots, correlation heatmaps)
- Comparison of Logistic Regression, KNN, and Random Forest
- Complete model evaluation with ROC curves and confusion matrices

### `Heart_Disease_Classifier.ipynb`
Focused implementation showcasing:
- Clean, straightforward Random Forest workflow
- Step-by-step hyperparameter tuning process
- Clear comparison of baseline vs. optimized models

### `Heart_Disease_Classifier_Pipeline.ipynb`
Production-oriented implementation featuring:
- Scikit-learn Pipeline best practices
- Modular, maintainable code structure
- Prevention of data leakage during cross-validation
- Ready for deployment scenarios

## Notes

- Random seed (42) is set for reproducibility
- Cross-validation uses 5 folds (cv=5)
- Both RandomizedSearchCV and GridSearchCV use `refit=True` to automatically retrain the best model on the full training set

## License

This project is for educational purposes.

## Author

Created as part of a machine learning practice project.
