import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X_train = pd.read_csv("dataset/processed/X_train.csv")
y_train = pd.read_csv("dataset/processed/y_train.csv").values.ravel()

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

train_sizes, train_scores, val_scores = learning_curve(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

os.makedirs("reports", exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, marker="o", label="Training Accuracy")
plt.plot(train_sizes, val_mean, marker="o", label="Validation Accuracy")
plt.title("Learning Curve: Overfitting Analysis")
plt.xlabel("Training Data Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reports/learning_curve.png")
plt.close()

print("Learning curve saved: reports/learning_curve.png")
print("Training Accuracy:", round(train_mean[-1] * 100, 2), "%")
print("Validation Accuracy:", round(val_mean[-1] * 100, 2), "%")