import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("dataset/raw/android_runtime_15000.csv")

df = df.drop_duplicates()
df = df.fillna(df.mean(numeric_only=True))

X = df.drop("Crash", axis=1)
y = df["Crash"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("dataset/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(
    "dataset/processed/X_train.csv",
    index=False
)

pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(
    "dataset/processed/X_test.csv",
    index=False
)

y_train.to_csv("dataset/processed/y_train.csv", index=False)
y_test.to_csv("dataset/processed/y_test.csv", index=False)

joblib.dump(scaler, "models/scaler.pkl")

print("Preprocessing completed successfully!")
print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])