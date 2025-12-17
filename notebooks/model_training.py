# Install Required Libraries
!pip install pandas scikit-learn matplotlib seaborn

# Import Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

print("All libraries imported successfully")

# Load the Data
df = pd.read_csv('data/raw/weatherAUS.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
df.head()

# Drop Missing Values
# Initial data inspection
print("\nMissing values per column:")
print(df.count())

print("\nData types:")
print(df.info())

print("\nColumn names:")
print(df.columns)

# Remove rows with any missing values
df_clean = df.dropna()

print(f"Rows removed: {len(df) - len(df_clean)}")
print(f"Clean dataset shape: {df_clean.shape}")
df = df_clean

# Adjust column names to reflect the prediction task
# RainToday: whether it rained today (TARGET variable)
# RainYesterday: whether it rained yesterday (FEATURE)
df = df.rename(columns={
    'RainToday': 'RainYesterday',
    'RainTomorrow': 'RainToday'
})

print("Columns renamed:")
print("RainTomorrow → RainToday (target to predict)")
print("RainToday → RainYesterday (historical feature)")

# Focus on nearby Melbourne locations for consistency
melbourne_locations = ['Melbourne', 'MelbourneAirport', 'Watsonia']
df = df[df.Location.isin(melbourne_locations)]

print(f"\nFiltered to Melbourne region")
print(f"Locations: {df['Location'].unique()}")
print(f"Dataset shape: {df.shape}")

# Define function to convert date to season
def date_to_season(date):
    """Convert month to season"""
    month = date.month
    if month in [12, 1, 2]:
        return 'Summer'
    elif month in [3, 4, 5]:
        return 'Autumn'
    elif month in [6, 7, 8]:
        return 'Winter'
    elif month in [9, 10, 11]:
        return 'Spring'

# Convert Date column to datetime and create Season feature
df['Date'] = pd.to_datetime(df['Date'])
df['Season'] = df['Date'].apply(date_to_season)

# Remove Date column (no longer needed)
df = df.drop(columns=['Date'])

print("Season feature created")
print(f"\nSeason distribution:")
print(df['Season'].value_counts())

# Check class balance
print("Target variable distribution (RainToday):")
print(df['RainToday'].value_counts())
print("\nClass proportions:")
print(df['RainToday'].value_counts(normalize=True))

# Visualize class balance
plt.figure(figsize=(8, 5))
df['RainToday'].value_counts().plot(kind='bar', color=['skyblue', 'orange'])
plt.title('Distribution of Target Variable (RainToday)')
plt.xlabel('Rain Today')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Show dataset info after cleaning
print("\nFinal dataset info:")
print(df.info())
print("\nDataset preview:")
print(df.head())

# Define features (X) and target (y)
X=df.drop(columns=['RainToday'], axis=1)  # All columns except target
y=df['RainToday']  # Target column

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeature columns: {list(X.columns)}")

# Split data into train and test sets
X_train, X_test, y_train, y_test=train_test_split(
    X, y, 
    test_size=0.2,           # 80% train, 20% test
    stratify=y,               # Maintain class proportions
    random_state=42
)

print(f"Training set: {X_train.shape} samples")
print(f"Test set: {X_test.shape} samples")
print(f"\nTraining set class distribution:")
print(y_train.value_counts(normalize=True))

# Automatically detect feature types
numeric_features=X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features=X_train.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nNumeric features ({len(numeric_features)}):")
print(numeric_features)
print(f"\nCategorical features ({len(categorical_features)}):")
print(categorical_features)

# Define preprocessing for numeric features
numeric_transformer=Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical features
categorical_transformer=Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

print("Preprocessing pipelines defined")

# Combine preprocessing steps
preprocessor=ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

print("Column transformer created")
print(f"Numeric: StandardScaler")
print(f"Categorical: OneHotEncoder")

# Create ML pipeline: Preprocessing to Random Forest
pipeline_rf=Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

print("Random Forest pipeline created")

# Define parameters to tune
param_grid_rf={
    'classifier__n_estimators': [50, 100],        # Number of trees
    'classifier__max_depth': [None, 10, 20],      # Tree depth
    'classifier__min_samples_split': [2, 5]       # Minimum samples to split
}

print("Parameter combinations to test:", 
      len(param_grid_rf['classifier__n_estimators']) * 
      len(param_grid_rf['classifier__max_depth']) * 
      len(param_grid_rf['classifier__min_samples_split']))

# Setup cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create GridSearchCV object
grid_search_rf=GridSearchCV(
    pipeline_rf,
    param_grid_rf,
    cv=cv,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1  # Use all processors
)

print("GridSearchCV configured with 5-fold stratified cross-validation")

# Fit the model
print("\nTraining Random Forest...")
grid_search_rf.fit(X_train, y_train)

print("\n"+"="*60)
print("RANDOM FOREST - BEST RESULTS")
print("="*60)

# Best parameters
print(f"\nBest parameters found:")
for param, value in grid_search_rf.best_params_.items():
    print(f"  {param}: {value}")

# Best CV score
print(f"\nBest cross-validation score: {grid_search_rf.best_score_:.4f}")

# Test set score
test_score_rf=grid_search_rf.score(X_test, y_test)
print(f"Test set accuracy: {test_score_rf:.4f}")

# Make predictions
y_pred_rf=grid_search_rf.predict(X_test)

# Classification report
print("\n"+"-"*60)
print("RANDOM FOREST - DETAILED METRICS")
print("-"*60)
print(classification_report(y_test, y_pred_rf))

# Generate and display confusion matrix
conf_matrix_rf=confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Random Forest - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Extract feature importances
feature_importances_rf=grid_search_rf.best_estimator_['classifier'].feature_importances_

# Combine feature names
feature_names=numeric_features+list(
    grid_search_rf.best_estimator_['preprocessor']
    .named_transformers_['cat']
    .named_steps['onehot']
    .get_feature_names_out(categorical_features)
)

# Create importance dataframe
importance_df_rf=pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances_rf
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance_df_rf.head(10))

# Plot top N features
N = 15
top_features_rf=importance_df_rf.head(N)

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_features_rf)), top_features_rf['Importance'], color='steelblue')
plt.yticks(range(len(top_features_rf)), top_features_rf['Feature'])
plt.gca().invert_yaxis()
plt.xlabel('Importance Score')
plt.title(f'Random Forest - Top {N} Most Important Features')
plt.tight_layout()
plt.show()

# Create ML pipeline: Preprocessing to Logistic Regression
pipeline_lr=Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

print("Logistic Regression pipeline created")

# Define parameters to tune for Logistic Regression
param_grid_lr={
    'classifier__solver': ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}

print("Parameter combinations to test:", 
      len(param_grid_lr['classifier__solver']) * 
      len(param_grid_lr['classifier__penalty']) * 
      len(param_grid_lr['classifier__class_weight']))

# Create GridSearchCV object
grid_search_lr=GridSearchCV(
    pipeline_lr,
    param_grid_lr,
    cv=cv,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

print("GridSearchCV configured for Logistic Regression")

# Fit the model
print("\nTraining Logistic Regression...")
grid_search_lr.fit(X_train, y_train)

print("\n"+"="*60)
print("LOGISTIC REGRESSION - BEST RESULTS")
print("="*60)

# Best parameters
print(f"\nBest parameters found:")
for param, value in grid_search_lr.best_params_.items():
    print(f"  {param}: {value}")

# Best CV score
print(f"\nBest cross-validation score: {grid_search_lr.best_score_:.4f}")

# Test set score
test_score_lr=grid_search_lr.score(X_test, y_test)
print(f"Test set accuracy: {test_score_lr:.4f}")

# Make predictions
y_pred_lr=grid_search_lr.predict(X_test)

# Classification report
print("\n"+"-"*60)
print("LOGISTIC REGRESSION - DETAILED METRICS")
print("-"*60)
print(classification_report(y_test, y_pred_lr))

# Generate and display confusion matrix
conf_matrix_lr=confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_lr, annot=True, fmt='d', cmap='Greens', cbar=True)
plt.title('Logistic Regression - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Create comparison dataframe
comparison_df=pd.DataFrame({
    'Model': ['Random Forest', 'Logistic Regression'],
    'CV Score': [grid_search_rf.best_score_, grid_search_lr.best_score_],
    'Test Accuracy': [test_score_rf, test_score_lr]
})

print("\n"+"="*60)
print("MODEL COMPARISON")
print("="*60)
print(comparison_df)

print("\n"+"-"*60)
print("WINNER:")
print("-"*60)
best_model_name=comparison_df.loc[comparison_df['Test Accuracy'].idxmax(), 'Model']
best_accuracy=comparison_df['Test Accuracy'].max()
print(f"{best_model_name}: {best_accuracy:.4f} accuracy")

# Plot comparison
fig, axes=plt.subplots(1, 2, figsize=(12, 5))

# CV Scores
axes.bar(comparison_df['Model'], comparison_df['CV Score'], color=['steelblue', 'green'])
axes.set_title('Cross-Validation Scores')
axes.set_ylabel('CV Score')
axes.set_ylim([0.7, 0.9])
for i, v in enumerate(comparison_df['CV Score']):
    axes.text(i, v + 0.01, f'{v:.4f}', ha='center')

# Test Accuracy
axes.bar(comparison_df['Model'], comparison_df['Test Accuracy'], color=['steelblue', 'green'])
axes.set_title('Test Set Accuracy')
axes.set_ylabel('Accuracy')
axes.set_ylim([0.7, 0.9])
for i, v in enumerate(comparison_df['Test Accuracy']):
    axes.text(i, v + 0.01, f'{v:.4f}', ha='center')

plt.tight_layout()
plt.show()

# Summary

print("\n" + "="*60)
print("PROJECT SUMMARY")
print("="*60)

print(f"\n1. BEST MODEL: {best_model_name}")
print(f"   - Test Accuracy: {best_accuracy:.2%}")

print(f"\n2. DATASET:")
print(f"   - Total samples: {len(df)}")
print(f"   - Training samples: {len(X_train)}")
print(f"   - Test samples: {len(X_test)}")

print(f"\n3. MOST IMPORTANT FEATURE:")
print(f"   - {importance_df_rf.iloc['Feature']}: {importance_df_rf.iloc['Importance']:.4f}")

print(f"\n4. HYPERPARAMETER TUNING:")
print(f"   - Total configurations tested: 12 (Random Forest) + 6 (Logistic Regression)")
print(f"   - Cross-validation folds: 5")

print("\n5. RECOMMENDATIONS:")
print(f"   - Use {best_model_name} for production deployment")
print(f"   - Model can predict rainfall with ~{best_accuracy:.0%} accuracy")
print(f"   - Focus on {importance_df_rf.iloc['Feature']} for data collection")
