import streamlit as st
import pandas as pd
import pickle
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, SRC_PATH)

from self_healing_engine import self_healing_recommendation
from risk_index import calculate_acri, calculate_dhi, get_risk_level
from root_cause_ranking import root_cause_ranking

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hybrid_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)

# Load model and scaler
model = joblib.load("models/hybrid_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(
    page_title="Android Crash Prediction",
    layout="wide"
)

st.title("Adaptive Android Crash Prediction and Self-Healing Framework")
st.write("Runtime telemetry-based Android crash prediction with ACRI, DHI, root cause ranking, and self-healing recommendations.")

# Input section
col1, col2 = st.columns(2)

with col1:
    cpu_usage = st.slider("CPU Usage (%)", 0, 100, 50)
    ram_usage = st.slider("RAM Usage (%)", 0, 100, 50)
    battery_level = st.slider("Battery Level (%)", 0, 100, 60)
    battery_temperature = st.slider("Battery Temperature (°C)", 20, 60, 35)
    fps = st.slider("FPS", 10, 120, 60)

with col2:
    storage_usage = st.slider("Storage Usage (%)", 0, 100, 50)
    network_latency = st.slider("Network Latency (ms)", 0, 500, 100)
    active_threads = st.slider("Active Threads", 0, 300, 80)
    app_launch_time = st.slider("App Launch Time (ms)", 300, 6000, 1000)
    device_temperature = st.slider("Device Temperature (°C)", 20, 65, 35)

input_data = {
    "CPU_Usage": cpu_usage,
    "RAM_Usage": ram_usage,
    "Battery_Level": battery_level,
    "Battery_Temperature": battery_temperature,
    "FPS": fps,
    "Storage_Usage": storage_usage,
    "Network_Latency": network_latency,
    "Active_Threads": active_threads,
    "App_Launch_Time": app_launch_time,
    "Device_Temperature": device_temperature
}

df = pd.DataFrame([input_data])

if st.button("Predict Crash Risk"):

    # Scale input data
    scaled_data = scaler.transform(df)

    # ML prediction
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    st.subheader("Machine Learning Prediction Result")

    if prediction == 1:
        st.error("Crash Risk Detected")
    else:
        st.success("No Crash Risk Detected")

    st.metric("Crash Probability", f"{probability * 100:.2f}%")

    if probability >= 0.75:
        ml_risk_level = "HIGH"
    elif probability >= 0.50:
        ml_risk_level = "MEDIUM"
    else:
        ml_risk_level = "LOW"

    st.metric("ML Risk Level", ml_risk_level)

    # Novelty indices
    acri = calculate_acri(input_data)
    dhi = calculate_dhi(acri)
    acri_level = get_risk_level(acri)

    st.subheader("Novelty-Based Risk Indices")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric("ACRI", f"{acri}/100")

    with col4:
        st.metric("DHI", f"{dhi}/100")

    with col5:
        st.metric("ACRI Risk Level", acri_level)

    # Root cause ranking
    st.subheader("Root Cause Ranking")

    causes = root_cause_ranking(input_data)

    for i, cause in enumerate(causes, 1):
        st.write(f"{i}. {cause[0]} - Score: {round(cause[1], 2)}")

    # Self-healing
    st.subheader("Self-Healing Recommendations")

    recommendations = self_healing_recommendation(input_data)

    for rec in recommendations:
        st.write("- " + rec)

    # Input data
    st.subheader("Input Runtime Data")
    st.dataframe(df)
