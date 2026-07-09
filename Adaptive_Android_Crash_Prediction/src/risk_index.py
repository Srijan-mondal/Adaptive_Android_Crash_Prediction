def calculate_acri(data):
    score = 0

    score += data["CPU_Usage"] * 0.15
    score += data["RAM_Usage"] * 0.15
    score += (100 - data["Battery_Level"]) * 0.10
    score += data["Battery_Temperature"] * 0.12
    score += (120 - data["FPS"]) * 0.12
    score += data["Storage_Usage"] * 0.10
    score += min(data["Network_Latency"] / 5, 100) * 0.08
    score += min(data["Active_Threads"] / 3, 100) * 0.08
    score += min(data["App_Launch_Time"] / 60, 100) * 0.05
    score += data["Device_Temperature"] * 0.05

    return round(min(score, 100), 2)


def calculate_dhi(acri):
    dhi = 100 - acri
    return round(max(dhi, 0), 2)


def get_risk_level(acri):
    if acri >= 75:
        return "HIGH"
    elif acri >= 50:
        return "MEDIUM"
    else:
        return "LOW"


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

    acri = calculate_acri(sample)
    dhi = calculate_dhi(acri)

    print("Adaptive Crash Risk Index:", acri)
    print("Device Health Index:", dhi)
    print("Risk Level:", get_risk_level(acri))