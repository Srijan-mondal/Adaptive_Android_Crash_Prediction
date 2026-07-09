import pandas as pd
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

X_train = pd.read_csv("dataset/processed/X_train.csv")
X_test = pd.read_csv("dataset/processed/X_test.csv")

y_train = pd.read_csv("dataset/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("dataset/processed/y_test.csv").values.ravel()

logistic = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

random_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

gradient_boosting = GradientBoostingClassifier(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

hybrid_model = VotingClassifier(
    estimators=[
        ("Logistic Regression", logistic),
        ("Random Forest", random_forest),
        ("Gradient Boosting", gradient_boosting)
    ],
    voting="soft"
)

hybrid_model.fit(X_train, y_train)

y_pred = hybrid_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Hybrid Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

cv_scores = cross_val_score(
    hybrid_model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\n5-Fold Cross Validation Accuracy:")
print(cv_scores)
print("Average CV Accuracy:", round(cv_scores.mean() * 100, 2), "%")

os.makedirs("models", exist_ok=True)
joblib.dump(hybrid_model, "models/hybrid_model.pkl")

print("\nModel saved successfully!")
