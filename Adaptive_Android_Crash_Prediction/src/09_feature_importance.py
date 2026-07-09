import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

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

model = joblib.load("models/hybrid_model.pkl")

# VotingClassifier model list:
# 0 = Logistic Regression
# 1 = Random Forest
# 2 = Gradient Boosting
random_forest = model.named_estimators_["Random Forest"]

importance = random_forest.feature_importances_

df_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

os.makedirs("reports", exist_ok=True)

df_importance.to_csv("reports/feature_importance.csv", index=False)

plt.figure(figsize=(8, 5))
plt.barh(df_importance["Feature"], df_importance["Importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("reports/feature_importance.png")
plt.close()

print("Feature importance saved successfully.")
print(df_importance)