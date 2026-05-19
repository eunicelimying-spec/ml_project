# =========================
# MILESTONE 2: ARCHITECTURE LOGIC
# Student Performance Prediction (Pass/Fail)
# Fixed Version - No Data Leakage
# =========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=== Milestone 2: Architecture Logic (Fixed Version) ===\n")

# =========================
# STEP 1: Load Dataset
# =========================
df = pd.read_csv("StudentPerformanceFactors.csv")

print("=== Dataset Shape ===")
print(df.shape)

# =========================
# STEP 2: Handle Missing Values
# =========================
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

print("\n=== Missing Values After Cleaning ===")
print(df.isnull().sum())

# =========================
# STEP 3: Create Target Variable
# =========================
# Pass = 1 if Exam_Score >= 70, otherwise Fail = 0
df['Pass_Fail'] = df['Exam_Score'].apply(lambda x: 1 if x >= 70 else 0)

print("\n=== Pass/Fail Distribution ===")
print(df['Pass_Fail'].value_counts(normalize=True))

# =========================
# STEP 4: Encode Categorical Variables
# =========================
df_encoded = pd.get_dummies(df, drop_first=True)
df_encoded = df_encoded.astype(int)

# Remove Exam_Score to prevent data leakage
X = df_encoded.drop(['Pass_Fail', 'Exam_Score'], axis=1)
y = df_encoded['Pass_Fail']

print(f"\nFeatures used for training: {X.shape[1]} columns (Exam_Score removed)")

# =========================
# STEP 5: Data Splitting
# Train 80%, Validation 10%, Test 10%
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("\n=== Data Split Summary ===")
print(f"Total samples: {len(df_encoded)}")
print(f"Training set  : {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Testing set   : {X_test.shape}")

# =========================
# STEP 6: Model Architecture Logic
# =========================
print("\n=== Why These Models? ===")
print("""
We chose:
1. Logistic Regression -> Simple, fast, and interpretable baseline model.
2. Random Forest Classifier -> Main model because:
   - Works very well with tabular datasets
   - Handles non-linear relationships
   - Provides feature importance
   - Robust to overfitting
   - Commonly used for student performance prediction
""")

# =========================
# STEP 7: Train Models
# =========================
print("\n=== Training Models ===")

# Logistic Regression (Baseline)
log_reg = LogisticRegression(
    max_iter=1000,
    random_state=42
)
log_reg.fit(X_train, y_train)

# Random Forest (Main Model)
rf_clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train, y_train)

# =========================
# STEP 8: Validation Evaluation
# =========================
y_val_pred_lr = log_reg.predict(X_val)
y_val_pred_rf = rf_clf.predict(X_val)

print("\n=== Logistic Regression Validation Results ===")
print("Accuracy:", round(accuracy_score(y_val, y_val_pred_lr), 4))
print(classification_report(y_val, y_val_pred_lr))

print("\n=== Random Forest Validation Results ===")
print("Accuracy:", round(accuracy_score(y_val, y_val_pred_rf), 4))
print(classification_report(y_val, y_val_pred_rf))

# =========================
# STEP 9: Feature Importance
# =========================
print("\n=== Top 10 Most Important Features (Random Forest) ===")

importances = pd.Series(
    rf_clf.feature_importances_,
    index=X_train.columns
)

top10 = importances.sort_values(ascending=False).head(10)
print(top10)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(
    x=top10.values,
    y=top10.index,
    palette="viridis"
)
plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# =========================
# STEP 10: Confusion Matrix
# =========================
plt.figure(figsize=(6, 5))

sns.heatmap(
    confusion_matrix(y_val, y_val_pred_rf),
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Fail (0)', 'Pass (1)'],
    yticklabels=['Fail (0)', 'Pass (1)']
)

plt.title("Confusion Matrix - Random Forest (Validation Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# COMPLETION MESSAGE
# =========================
print("\nMilestone 2 Completed Successfully!")
print("Data leakage has been fixed by removing 'Exam_Score'.")
print("Ready for Milestone 3 (Training Loop).")
