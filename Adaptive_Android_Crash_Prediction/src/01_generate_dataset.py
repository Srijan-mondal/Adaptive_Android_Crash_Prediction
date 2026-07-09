import pandas as pd
import numpy as np
import os

np.random.seed(42)

n = 15000

cpu_usage = np.random.uniform(10, 100, n)
ram_usage = np.random.uniform(15, 100, n)
battery_level = np.random.randint(5, 101, n)
battery_temperature = np.random.uniform(25, 55, n)
fps = np.random.randint(10, 121, n)
storage_usage = np.random.uniform(20, 100, n)
network_latency = np.random.randint(10, 501, n)
active_threads = np.random.randint(20, 301, n)
app_launch_time = np.random.randint(300, 6001, n)
device_temperature = np.random.uniform(28, 60, n)

risk_score = (
    cpu_usage * 0.18 +
    ram_usage * 0.18 +
    (100 - battery_level) * 0.10 +
    battery_temperature * 0.12 +
    (120 - fps) * 0.14 +
    storage_usage * 0.10 +
    network_latency * 0.05 +
    active_threads * 0.06 +
    app_launch_time * 0.04 +
    device_temperature * 0.13
)

risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())

crash = (risk_score > 0.55).astype(int)

noise = np.random.binomial(1, 0.02, n)
crash = np.where(noise == 1, 1 - crash, crash)

df = pd.DataFrame({
    "CPU_Usage": cpu_usage.round(2),
    "RAM_Usage": ram_usage.round(2),
    "Battery_Level": battery_level,
    "Battery_Temperature": battery_temperature.round(2),
    "FPS": fps,
    "Storage_Usage": storage_usage.round(2),
    "Network_Latency": network_latency,
    "Active_Threads": active_threads,
    "App_Launch_Time": app_launch_time,
    "Device_Temperature": device_temperature.round(2),
    "Crash": crash
})

os.makedirs("dataset/raw", exist_ok=True)
df.to_csv("dataset/raw/android_runtime_15000.csv", index=False)

print("Dataset created successfully!")
print(df.head())
print("\nClass distribution:")
print(df["Crash"].value_counts())