# =========================
# MILESTONE 4: MODEL OPTIMIZATION
# =========================

# Imports
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

print("\n=== Milestone 4: Model Optimization ===")

# Evaluate model before optimization
before_train_pred = final_model.predict(X_train)
before_val_pred = final_model.predict(X_val)

before_train_acc = accuracy_score(y_train, before_train_pred)
before_val_acc = accuracy_score(y_val, before_val_pred)

print("\n=== Before Optimization ===")
print(f"Training Accuracy: {before_train_acc:.4f}")
print(f"Validation Accuracy: {before_val_acc:.4f}")
print(f"Overfitting Gap: {before_train_acc - before_val_acc:.4f}")

# Hyperparameter tuning grid
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}

# Grid Search
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=1),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=1
)

grid_search.fit(X_train, y_train)

print("\n=== Best Parameters ===")
print(grid_search.best_params_)

# Optimized model
optimized_model = grid_search.best_estimator_

# Evaluate optimized model
after_train_pred = optimized_model.predict(X_train)
after_val_pred = optimized_model.predict(X_val)

after_train_acc = accuracy_score(y_train, after_train_pred)
after_val_acc = accuracy_score(y_val, after_val_pred)

print("\n=== After Optimization ===")
print(f"Training Accuracy: {after_train_acc:.4f}")
print(f"Validation Accuracy: {after_val_acc:.4f}")
print(f"Overfitting Gap: {after_train_acc - after_val_acc:.4f}")

print("\n=== Optimized Model Validation Report ===")
print(classification_report(y_val, after_val_pred))

# Comparison table
comparison = pd.DataFrame({
    'Model': ['Before Optimization', 'After Optimization'],
    'Training Accuracy': [before_train_acc, after_train_acc],
    'Validation Accuracy': [before_val_acc, after_val_acc],
    'Overfitting Gap': [
        before_train_acc - before_val_acc,
        after_train_acc - after_val_acc
    ]
})

print("\n=== Before vs After Optimization Comparison ===")
print(comparison)

# Visualization
comparison.plot(
    x='Model',
    y=['Training Accuracy', 'Validation Accuracy', 'Overfitting Gap'],
    kind='bar'
)

plt.title("Before vs After Model Optimization")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.show()