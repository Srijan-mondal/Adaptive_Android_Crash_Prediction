def self_healing_recommendation(data):
    recommendations = []

    if data["CPU_Usage"] > 85:
        recommendations.append("CPU overload detected: reduce background processes.")

    if data["RAM_Usage"] > 85:
        recommendations.append("High RAM usage detected: clear cache and optimize memory.")

    if data["Battery_Level"] < 20:
        recommendations.append("Low battery detected: enable battery saver mode.")

    if data["Battery_Temperature"] > 45:
        recommendations.append("Battery overheating detected: activate thermal protection.")

    if data["FPS"] < 25:
        recommendations.append("Low FPS detected: reduce graphics rendering load.")

    if data["Storage_Usage"] > 90:
        recommendations.append("High storage usage detected: delete temporary files.")

    if data["Network_Latency"] > 350:
        recommendations.append("High network latency detected: retry connection with backoff.")

    if data["Active_Threads"] > 220:
        recommendations.append("Too many active threads: restart idle background services.")

    if data["App_Launch_Time"] > 4500:
        recommendations.append("Slow app launch detected: optimize startup services.")

    if data["Device_Temperature"] > 50:
        recommendations.append("Device overheating detected: reduce workload temporarily.")

    if not recommendations:
        recommendations.append("Device condition is stable. No healing action required.")

    return recommendations


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

    result = self_healing_recommendation(sample)

    print("Self-Healing Recommendations:")
    for i, rec in enumerate(result, 1):
        print(f"{i}. {rec}")