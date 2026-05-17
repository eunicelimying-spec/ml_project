import pandas as pd
import matplotlib.pyplot as plt

# =========================
# STEP 1: Load Dataset
# =========================
df = pd.read_csv("StudentPerformanceFactors.csv")

print("=== First 5 Rows ===")
print(df.head())

# =========================
# STEP 2: Understand Dataset
# =========================
print("\n=== Dataset Shape ===")
print(df.shape)

print("\n=== Dataset Info ===")
print(df.info())

print("\n=== Statistical Summary ===")
print(df.describe())

# =========================
# STEP 3: Check Missing Values
# =========================
print("\n=== Missing Values Before Cleaning ===")
print(df.isnull().sum())

# =========================
# Handle Missing Values
# =========================

# Fill categorical columns with mode
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])

# Fill numerical columns with mean
for col in df.columns:
    if df[col].dtype != 'object':
        df[col] = df[col].fillna(df[col].mean())

print("\n=== Missing Values After Cleaning ===")
print(df.isnull().sum())

# =========================
# STEP 4: Create Target Variable
# =========================

df['Pass_Fail'] = df['Exam_Score'].apply(lambda x: 1 if x >= 70 else 0)

print("\n=== Target Variable (Pass/Fail) ===")
print(df[['Exam_Score', 'Pass_Fail']].head())

print("\n=== Pass/Fail Distribution ===")
print(df['Pass_Fail'].value_counts())

# =========================
# EDA VISUALIZATION 
# =========================

# 1. Exam Score Distribution
plt.figure()
df['Exam_Score'].hist()
plt.title("Distribution of Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("Frequency")
plt.show()

# 2. Pass vs Fail Distribution
plt.figure()
df['Pass_Fail'].value_counts().plot(kind='bar')
plt.title("Pass vs Fail Distribution")
plt.xlabel("Pass_Fail (0 = Fail, 1 = Pass)")
plt.ylabel("Count")
plt.show()

# =========================
# Encode Categorical Variables
# =========================

df_encoded = pd.get_dummies(df, drop_first=True)

df_encoded = df_encoded.astype(int)

print("\n=== Encoded Dataset Preview ===")
print(df_encoded.head())

print("\n=== Dataset Shape After Encoding ===")
print(df_encoded.shape)

# =========================
# STEP 5: Data Splitting
# =========================

from sklearn.model_selection import train_test_split

# Features (X) and Target (y)
X = df_encoded.drop('Pass_Fail', axis=1)
y = df_encoded['Pass_Fail']

# First split: Train (80%) and Temp (20%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: Validation (10%) and Test (10%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# =========================
# Display Results
# =========================

print("\n=== Data Split Summary ===")
print(f"Total samples: {len(df_encoded)}")

print("\nTraining set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Testing set:", X_test.shape)

# Check distribution
print("\n=== Target Distribution ===")
print("Train:\n", y_train.value_counts(normalize=True))
print("Validation:\n", y_val.value_counts(normalize=True))
print("Test:\n", y_test.value_counts(normalize=True))

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
df = pd.read_csv("StudentPerformanceFactors.csv")

print("=== Dataset Shape ===")
print(df.shape)

# Handle Missing Values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

print("\n=== Missing Values After Cleaning ===")
print(df.isnull().sum())

# Create Target Variable (Binary: Pass/Fail)
df['Pass_Fail'] = df['Exam_Score'].apply(lambda x: 1 if x >= 70 else 0)

print("\n=== Pass/Fail Distribution ===")
print(df['Pass_Fail'].value_counts(normalize=True))

# Encode Categorical Variables
df_encoded = pd.get_dummies(df, drop_first=True)
df_encoded = df_encoded.astype(int)

X = df_encoded.drop(['Pass_Fail', 'Exam_Score'], axis=1)   # ← Fixed here
y = df_encoded['Pass_Fail']

print(f"\nFeatures used for training: {X.shape[1]} columns (Exam_Score removed)")

# =========================
# Data Splitting (Same as Milestone 1)
# =========================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("\n=== Data Split Summary ===")
print(f"Total samples: {len(df_encoded)}")
print(f"Training set  : {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Testing set   : {X_test.shape}")

# =========================
# MODEL ARCHITECTURE LOGIC
# =========================
print("\n=== Why These Models? ===")
print("""
We chose:
1. Logistic Regression → Simple, fast, interpretable baseline (linear model).
2. Random Forest Classifier → Main model because:
   - Excellent for tabular data (mix of numerical + categorical features)
   - Handles non-linear relationships and feature interactions well
   - Built-in feature importance (helps explain which student factors matter most)
   - Robust to overfitting through ensemble of trees
   - Commonly used and performs strongly on student performance datasets
""")

# =========================
# Train Models
# =========================
print("\n=== Training Models ===")

# Logistic Regression (Baseline)
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# Random Forest (Main Model)
rf_clf = RandomForestClassifier(
    n_estimators=100, 
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train, y_train)

# =========================
# Predictions & Evaluation on Validation Set
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
# Feature Importance (Random Forest)
# =========================
print("\n=== Top 10 Most Important Features (Random Forest) ===")
importances = pd.Series(rf_clf.feature_importances_, index=X_train.columns)
top10 = importances.sort_values(ascending=False).head(10)
print(top10)

# Plot Top 10 Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x=top10.values, y=top10.index, palette="viridis")
plt.title("Top 10 Feature Importances - Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# =========================
# Confusion Matrix for Random Forest
# =========================
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_val, y_val_pred_rf), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fail (0)', 'Pass (1)'],
            yticklabels=['Fail (0)', 'Pass (1)'])
plt.title("Confusion Matrix - Random Forest (Validation Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nMilestone 2 Completed Successfully!")
print("Data leakage has been fixed by removing 'Exam_Score'.")
print("Ready for Milestone 3 (Training Loop).")
