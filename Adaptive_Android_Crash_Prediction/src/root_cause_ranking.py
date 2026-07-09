def root_cause_ranking(data):
    causes = []

    causes.append(("CPU Overload", data["CPU_Usage"]))
    causes.append(("RAM Pressure", data["RAM_Usage"]))
    causes.append(("Low Battery Risk", 100 - data["Battery_Level"]))
    causes.append(("Battery Heating", data["Battery_Temperature"] * 2))
    causes.append(("FPS Drop", 120 - data["FPS"]))
    causes.append(("Storage Pressure", data["Storage_Usage"]))
    causes.append(("Network Delay", data["Network_Latency"] / 5))
    causes.append(("Thread Overload", data["Active_Threads"] / 3))
    causes.append(("Slow App Launch", data["App_Launch_Time"] / 60))
    causes.append(("Device Heating", data["Device_Temperature"] * 2))

    causes = sorted(causes, key=lambda x: x[1], reverse=True)

    return causes[:5]


if __name__ == "__main__":
    sample = {
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
    }

    ranking = root_cause_ranking(sample)

    print("Top Root Causes:")
    for i, cause in enumerate(ranking, 1):
        print(f"{i}. {cause[0]} - Score: {round(cause[1], 2)}")