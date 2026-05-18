# =========================
# MILESTONE 3: TRAINING LOOP
# =========================

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

print("\n=== Milestone 3: Training Loop ===")

# Store accuracy results
train_acc_list = []
val_acc_list = []

# Different numbers of trees to test
estimators_range = range(10, 110, 10)

# Training loop
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
    
    # Accuracy scores
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    # Save results
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    
    # Print progress
    print(f"Trees={n} | Train Accuracy={train_acc:.4f} | Validation Accuracy={val_acc:.4f}")

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
# Select Best Model
# =========================

best_n = estimators_range[
    val_acc_list.index(max(val_acc_list))
]

print(f"\nBest n_estimators: {best_n}")

# Train final model
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

print("\nClassification Report:")
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