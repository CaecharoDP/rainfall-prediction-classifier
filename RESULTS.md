# Model Results & Performance

## Best Model Configuration
- **Algorithm**: Random Forest Classifier
- **Parameters**: 
  - n_estimators=xx
  - max_depth=xx
  - min_samples_split=xx

## Performance Metrics

### Test Set Results
| Metric | Score |
|--------|-------|
| Accuracy | xx% |
| Precision (??) | xx |
| Recall (??) | xx |
| F1-Score (??) | xx |

### Confusion Matrix
|          | Predicted No Rain | Predicted Rain |
|-------------------|---------|------|
| Actual No Rain           | xx   | xx   |
| Actual Rain              | xx     | xx  |

### Feature Importance (Top 10)
1. Humidity3pm - xx%
2. Pressure3pm - xx%
3. ?? - xx%
4. ??
5. ??
6. ??
7. ??
8. ??
9. ??
10. ??

## Model Comparison
| Model | Accuracy | F1-Score | Train Time |
|-------|----------|----------|-----------|
| Random Forest | xx% | xx | xxs |
| Logistic Regression | xx% | xx | xxs |

## Grid Search Results
- **CV Folds**: xx
- **Hyperparameters Tuned**: 
  - n_estimators: [xx, xxx]
  - max_depth: [xx, xxx, xxxx]
  - min_samples_split: [xx, xxx]
- **Total Configurations Tested**: xx
- **Best Score**: xx (CV average)

## Processing Pipeline
1. **Data Loading**: CSV from Kaggle
2. **Cleaning**: Handle missing values, drop irrelevant columns
3. **Preprocessing**: 
   - Numerical: StandardScaler
   - Categorical: OneHotEncoder
4. **Train/Test Split**: 80/20 stratified split
5. **Model Training**: GridSearchCV with 5-fold CV
6. **Evaluation**: Classification metrics & confusion matrix
