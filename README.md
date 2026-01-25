# Heart Disease Classifier

A machine learning project that predicts heart disease using Random Forest classification with hyperparameter optimization.

## Overview

This project builds and optimizes a Random Forest classifier to predict heart disease from patient data. The notebook explores three approaches:
- **Baseline Model**: Random Forest with default parameters
- **RandomizedSearchCV**: Efficient exploration of hyperparameter space
- **GridSearchCV**: Fine-tuned search for optimal parameters

## Dataset

The project uses `heart-disease.csv` containing patient health metrics with a binary target variable indicating heart disease presence.

## Features

- **Data preprocessing** and train/test split (80/20 ratio)
- **Baseline model** training with default Random Forest parameters
- **Hyperparameter optimization** using both RandomizedSearchCV and GridSearchCV
- **Comprehensive evaluation** using multiple metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- **Performance comparison** visualization across all three models

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
```

## Installation

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Usage

1. Ensure `heart-disease.csv` is in the same directory as the notebook
2. Open `Heart_Disease_Classifier.ipynb` in Jupyter Notebook or VS Code
3. Run all cells sequentially to:
   - Load and split the data
   - Train the baseline model
   - Perform hyperparameter tuning
   - Compare model performance

## Model Details

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

The project uses a custom `evaluate_preds()` function that computes and displays:
- Accuracy percentage
- Precision score
- Recall score
- F1 score

Results are compared visually using a bar chart showing all metrics across the three models.

## Results

The notebook compares all three approaches to identify the optimal configuration for heart disease prediction. Performance metrics are displayed both numerically and visually for easy comparison.

## Project Structure

```
Heart_Disease_Classifier/
├── Heart_Disease_Classifier.ipynb  # Main notebook
├── heart-disease.csv                # Dataset
└── README.md                        # This file
```

## Notes

- Random seed (42) is set for reproducibility
- Cross-validation uses 5 folds (cv=5)
- Both RandomizedSearchCV and GridSearchCV use `refit=True` to automatically retrain the best model on the full training set

## License

This project is for educational purposes.

## Author

Created as part of a machine learning practice project.
