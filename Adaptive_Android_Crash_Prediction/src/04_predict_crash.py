import pandas as pd
import joblib

model = joblib.load("models/hybrid_model.pkl")
scaler = joblib.load("models/scaler.pkl")

feature_names = [
    "CPU_Usage",
    "RAM_Usage",
    "Battery_Level",
    "Battery_Temperature",
    "FPS",
    "Storage_Usage",
    "Network_Latency",
    "Active_Threads",
    "App_Launch_Time",
    "Device_Temperature"
]

sample_data = pd.DataFrame([{
    "CPU_Usage": 92,
    "RAM_Usage": 88,
    "Battery_Level": 18,
    "Battery_Temperature": 47,
    "FPS": 22,
    "Storage_Usage": 94,
    "Network_Latency": 380,
    "Active_Threads": 240,
    "App_Launch_Time": 4800,
    "Device_Temperature": 52
}], columns=feature_names)

sample_scaled = scaler.transform(sample_data)

prediction = model.predict(sample_scaled)[0]
probability = model.predict_proba(sample_scaled)[0][1]

print("Crash Prediction:", "Crash Risk Detected" if prediction == 1 else "No Crash Risk")
print("Crash Probability:", round(probability * 100, 2), "%")

if probability >= 0.75:
    risk_level = "HIGH"
elif probability >= 0.50:
    risk_level = "MEDIUM"
else:
    risk_level = "LOW"

print("Risk Level:", risk_level)