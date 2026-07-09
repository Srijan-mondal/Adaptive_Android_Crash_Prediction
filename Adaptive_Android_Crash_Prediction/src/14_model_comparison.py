import pandas as pd
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

X_train = pd.read_csv("dataset/processed/X_train.csv")
X_test = pd.read_csv("dataset/processed/X_test.csv")
y_train = pd.read_csv("dataset/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("dataset/processed/y_test.csv").values.ravel()

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
}

hybrid_model = VotingClassifier(
    estimators=[
        ("lr", models["Logistic Regression"]),
        ("rf", models["Random Forest"]),
        ("gb", models["Gradient Boosting"])
    ],
    voting="soft"
)

models["Hybrid Voting Ensemble"] = hybrid_model

results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred) * 100, 2),
        "Recall": round(recall_score(y_test, y_pred) * 100, 2),
        "F1_Score": round(f1_score(y_test, y_pred) * 100, 2),
        "ROC_AUC": round(roc_auc_score(y_test, y_prob), 4)
    })

df_results = pd.DataFrame(results)

os.makedirs("reports", exist_ok=True)
df_results.to_csv("reports/model_comparison.csv", index=False)

print(df_results)
print("\nModel comparison saved: reports/model_comparison.csv")