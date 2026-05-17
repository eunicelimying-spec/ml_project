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

