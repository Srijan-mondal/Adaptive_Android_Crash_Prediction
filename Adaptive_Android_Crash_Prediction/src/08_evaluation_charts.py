import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

X_test = pd.read_csv("dataset/processed/X_test.csv")
y_test = pd.read_csv("dataset/processed/y_test.csv").values.ravel()

model = joblib.load("models/hybrid_model.pkl")

os.makedirs("reports", exist_ok=True)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Confusion Matrix
plt.figure(figsize=(6, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png")
plt.close()

# ROC Curve
plt.figure(figsize=(6, 4))
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve")
plt.tight_layout()
plt.savefig("reports/roc_curve.png")
plt.close()

print("Evaluation charts saved successfully.")