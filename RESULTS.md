# Model Results & Performance

## Best Model Configuration
- **Algorithm**: Random Forest Classifier
- **Parameters**: 
  - n_estimators=100
  - max_depth=20
  - min_samples_split=2

## Performance Metrics

### Test Set Results
| Metric | Score |
|--------|-------|
| Accuracy | 84.5% |
| Precision | 75% |
| Recall | 51% |
| F1-Score | 61% |

### Confusion Matrix
|          | Predicted No Rain | Predicted Rain |
|-------------------|---------|------|
| Actual No Rain           | 1094   | 60   |
| Actual Rain              | 175     | 183  |

## Model Comparison
| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Random Forest | 84.5% | 61% |
| Logistic Regression | 83% | 59% |

## Grid Search Results
- **CV Folds**: 5
- **Hyperparameters Tuned**: 
  - n_estimators: [50, 100]
  - max_depth: [None, 10, 20]
  - min_samples_split: [2, 5]
- **Total Configurations Tested**: 12
- **Best Score**: 85% (CV average)

## Processing Pipeline
1. **Data Loading**: CSV from Kaggle
2. **Cleaning**: Handle missing values, drop irrelevant columns
3. **Preprocessing**: 
   - Numerical: StandardScaler
   - Categorical: OneHotEncoder
4. **Train/Test Split**: 80/20 stratified split
5. **Model Training**: GridSearchCV with 5-fold CV
6. **Evaluation**: Classification metrics & confusion matrix
