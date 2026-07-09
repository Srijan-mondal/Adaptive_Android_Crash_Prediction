import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load data
X_train = pd.read_csv("dataset/processed/X_train.csv")
X_test = pd.read_csv("dataset/processed/X_test.csv")

y_train = pd.read_csv("dataset/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("dataset/processed/y_test.csv").values.ravel()

# Load trained model
model = joblib.load("models/hybrid_model.pkl")

# Training accuracy
train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)

# Test accuracy
test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)

# Overfitting gap
gap = train_accuracy - test_accuracy

print("=" * 50)
print("Training Accuracy :", round(train_accuracy * 100, 2), "%")
print("Testing Accuracy  :", round(test_accuracy * 100, 2), "%")
print("Overfitting Gap   :", round(gap * 100, 2), "%")
print("=" * 50)

if gap < 0.02:
    print("Result : Excellent Generalization (No Significant Overfitting)")
elif gap < 0.05:
    print("Result : Mild Overfitting")
elif gap < 0.10:
    print("Result : Moderate Overfitting")
else:
    print("Result : Severe Overfitting")