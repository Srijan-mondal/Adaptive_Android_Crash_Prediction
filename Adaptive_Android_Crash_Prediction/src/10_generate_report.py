import pandas as pd
import joblib
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

X_test = pd.read_csv("dataset/processed/X_test.csv")
y_test = pd.read_csv("dataset/processed/y_test.csv").values.ravel()

model = joblib.load("models/hybrid_model.pkl")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

os.makedirs("reports", exist_ok=True)

report_text = f"""
Adaptive Android Crash Prediction and Autonomous Self-Healing Framework

Project Summary:
This project predicts Android application crash risk using runtime telemetry features.
The system uses CPU usage, RAM usage, battery level, battery temperature, FPS,
storage usage, network latency, active threads, app launch time, and device temperature.

Model:
Hybrid Machine Learning Model using Logistic Regression, Random Forest,
and Gradient Boosting with Soft Voting Ensemble.

Performance Metrics:
Accuracy  : {accuracy * 100:.2f}%
Precision : {precision * 100:.2f}%
Recall    : {recall * 100:.2f}%
F1-Score  : {f1 * 100:.2f}%
ROC-AUC   : {auc:.4f}

Novelty:
1. Runtime telemetry-based Android crash prediction.
2. Adaptive crash probability estimation.
3. Device health-aware risk analysis.
4. Autonomous self-healing recommendation engine.
5. Explainable feature importance-based crash cause identification.

Self-Healing Examples:
- High CPU usage: reduce background processes.
- High RAM usage: memory optimization.
- Low FPS: graphics optimization.
- High temperature: thermal protection.
- High storage usage: cache cleanup.

Conclusion:
The proposed framework predicts crash risk before failure and recommends
preventive self-healing actions, making it useful for proactive Android
application reliability improvement.
"""

with open("reports/final_project_report.txt", "w") as file:
    file.write(report_text)

print("Final report generated: reports/final_project_report.txt")