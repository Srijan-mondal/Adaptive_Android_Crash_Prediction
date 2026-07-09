import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("dataset/raw/android_runtime_15000.csv")

os.makedirs("reports", exist_ok=True)

# Crash distribution
plt.figure(figsize=(6, 4))
df["Crash"].value_counts().plot(kind="bar")
plt.title("Crash vs No Crash Distribution")
plt.xlabel("Crash Class")
plt.ylabel("Count")
plt.xticks([0, 1], ["No Crash", "Crash"], rotation=0)
plt.tight_layout()
plt.savefig("reports/crash_distribution.png")
plt.close()

# CPU vs Crash
plt.figure(figsize=(6, 4))
df.boxplot(column="CPU_Usage", by="Crash")
plt.title("CPU Usage vs Crash")
plt.suptitle("")
plt.xlabel("Crash")
plt.ylabel("CPU Usage")
plt.tight_layout()
plt.savefig("reports/cpu_vs_crash.png")
plt.close()

# RAM vs Crash
plt.figure(figsize=(6, 4))
df.boxplot(column="RAM_Usage", by="Crash")
plt.title("RAM Usage vs Crash")
plt.suptitle("")
plt.xlabel("Crash")
plt.ylabel("RAM Usage")
plt.tight_layout()
plt.savefig("reports/ram_vs_crash.png")
plt.close()

# FPS vs Crash
plt.figure(figsize=(6, 4))
df.boxplot(column="FPS", by="Crash")
plt.title("FPS vs Crash")
plt.suptitle("")
plt.xlabel("Crash")
plt.ylabel("FPS")
plt.tight_layout()
plt.savefig("reports/fps_vs_crash.png")
plt.close()

print("EDA charts saved in reports folder.")