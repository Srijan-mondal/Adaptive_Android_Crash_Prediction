import pandas as pd
import joblib
import os

from risk_index import calculate_acri, calculate_dhi, get_risk_level
from root_cause_ranking import root_cause_ranking
from self_healing_engine import self_healing_recommendation

model = joblib.load("models/hybrid_model.pkl")
scaler = joblib.load("models/scaler.pkl")

input_file = "dataset/raw/android_runtime_15000.csv"
output_file = "reports/batch_prediction_result.csv"

df = pd.read_csv(input_file)

X = df.drop("Crash", axis=1)

X_scaled = scaler.transform(X)

predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)[:, 1]

df["Predicted_Crash"] = predictions
df["Crash_Probability"] = (probabilities * 100).round(2)

acri_values = []
dhi_values = []
risk_levels = []
root_causes = []
healing_actions = []

for _, row in X.iterrows():
    data = row.to_dict()

    acri = calculate_acri(data)
    dhi = calculate_dhi(acri)
    risk = get_risk_level(acri)

    causes = root_cause_ranking(data)
    healing = self_healing_recommendation(data)

    acri_values.append(acri)
    dhi_values.append(dhi)
    risk_levels.append(risk)
    root_causes.append(", ".join([c[0] for c in causes[:3]]))
    healing_actions.append(" | ".join(healing[:3]))

df["ACRI"] = acri_values
df["DHI"] = dhi_values
df["Risk_Level"] = risk_levels
df["Top_Root_Causes"] = root_causes
df["Self_Healing_Actions"] = healing_actions

os.makedirs("reports", exist_ok=True)
df.to_csv(output_file, index=False)

print("Batch prediction completed.")
print("Saved file:", output_file)
print(df.head())