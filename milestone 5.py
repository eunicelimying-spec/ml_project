import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("StudentPerformanceFactors.csv")

print("=== FIRST 5 ROWS ===")
print(df.head())

# =========================
# DATA CLEANING
# =========================

# Convert categorical columns using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Create target variable
df_encoded['Pass_Fail'] = df_encoded['Exam_Score'].apply(
    lambda x: 1 if x >= 70 else 0
)

# Features and target
X = df_encoded.drop(['Pass_Fail', 'Exam_Score'], axis=1)
y = df_encoded['Pass_Fail']

# =========================
# TRAIN / VALIDATION / TEST SPLIT
# =========================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

# =========================
# MODEL OPTIMIZATION
# =========================

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)

optimized_model = grid_search.best_estimator_

print("\n=== BEST PARAMETERS ===")
print(grid_search.best_params_)

# =========================
# FINAL EVALUATION
# =========================

y_test_pred = optimized_model.predict(X_test)

print("\n=== FINAL TEST RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_test_pred))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_test_pred))

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.show()

# =========================
# ROC CURVE
# =========================

y_probs = optimized_model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve.png")
plt.show()

# =========================
# ERROR ANALYSIS
# =========================

results = X_test.copy()

results['Actual'] = y_test.values
results['Predicted'] = y_test_pred

errors = results[results['Actual'] != results['Predicted']]

print("\n=== MISCLASSIFIED SAMPLES ===")
print(errors.head())

print(f"\nTotal Errors: {len(errors)}")

# =========================
# FEATURE IMPORTANCE
# =========================

importance = pd.Series(
    optimized_model.feature_importances_,
    index=X.columns
)

top10 = importance.sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))

sns.barplot(
    x=top10.values,
    y=top10.index
)

plt.title("Top 10 Important Features")

plt.savefig("feature_importance.png")
plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print("\nPrecision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("ROC-AUC:", roc_auc)

print("\nMilestone 5 Completed Successfully!")
