# =========================
# MILESTONE 3: TRAINING LOOP
# Fixed Version
# =========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\n=== Milestone 3: Training Loop ===")

# =========================
# Load Dataset
# =========================

df = pd.read_csv(
    r"C:\Users\admin\Downloads\Telegram Desktop\StudentPerformanceFactors.csv"
)

print("\n=== Dataset Loaded Successfully ===")
print(df.head())

# =========================
# Handle Missing Values
# =========================

for col in df.columns:

    # Numeric columns
    if pd.api.types.is_numeric_dtype(df[col]):

        df[col] = df[col].fillna(df[col].mean())

    # Categorical columns
    else:

        df[col] = df[col].fillna(df[col].mode()[0])

print("\n=== Missing Values After Cleaning ===")
print(df.isnull().sum())

# =========================
# Create Target Variable
# =========================

df['Pass_Fail'] = df['Exam_Score'].apply(
    lambda x: 1 if x >= 70 else 0
)

print("\n=== Pass/Fail Distribution ===")
print(df['Pass_Fail'].value_counts())

# =========================
# Encode Categorical Variables
# =========================

df_encoded = pd.get_dummies(df, drop_first=True)

# Convert boolean columns to integers
df_encoded = df_encoded.astype(int)

# =========================
# Features and Target
# =========================

X = df_encoded.drop(['Pass_Fail', 'Exam_Score'], axis=1)
y = df_encoded['Pass_Fail']

# =========================
# Split Dataset
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
print("Training Set :", X_train.shape)
print("Validation Set :", X_val.shape)
print("Testing Set :", X_test.shape)

# =========================
# Training Loop
# =========================

train_acc_list = []
val_acc_list = []

estimators_range = range(10, 110, 10)

for n in estimators_range:

    # Create model
    model = RandomForestClassifier(
        n_estimators=n,
        random_state=42,
        n_jobs=-1
    )

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    # Accuracy
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    # Save results
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

    # Print progress
    print(
        f"Trees={n} | "
        f"Train Accuracy={train_acc:.4f} | "
        f"Validation Accuracy={val_acc:.4f}"
    )

# =========================
# Plot Training Progress
# =========================

plt.figure(figsize=(8,5))

plt.plot(
    estimators_range,
    train_acc_list,
    marker='o',
    label='Train Accuracy'
)

plt.plot(
    estimators_range,
    val_acc_list,
    marker='o',
    label='Validation Accuracy'
)

plt.title("Training Progress")
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

# =========================
# Best Model Selection
# =========================

best_n = estimators_range[
    val_acc_list.index(max(val_acc_list))
]

print(f"\nBest n_estimators: {best_n}")

# =========================
# Train Final Model
# =========================

final_model = RandomForestClassifier(
    n_estimators=best_n,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train, y_train)

# =========================
# Final Testing
# =========================

y_test_pred = final_model.predict(X_test)

print("\n=== Final Test Results ===")

print(
    "Test Accuracy:",
    accuracy_score(y_test, y_test_pred)
)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_test_pred))

# =========================
# Confusion Matrix
# =========================

plt.figure(figsize=(6,4))

sns.heatmap(
    confusion_matrix(y_test, y_test_pred),
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

print("\nMilestone 3 Completed Successfully!")
