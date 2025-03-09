import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model & scaler
@st.cache_resource
def load_model():
    model = joblib.load("stress_model.pkl")  # Load trained model
    scaler = joblib.load("scaler.pkl")  # Load saved scaler
    return model, scaler

# Load model & scaler only once
model, scaler = load_model()

st.title("Stress Level Detection")

# User Input
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0)
step_count = st.number_input("Step Count", min_value=0, max_value=30000)

if st.button("Predict Stress Level"):
    # Scale user inputs
    input_data = np.array([[humidity, temperature, step_count]])
    input_scaled = scaler.transform(input_data)  # Apply the saved scaler

    # Make prediction
    prediction = model.predict(input_scaled)
    
    # Display Result
    stress_labels = {0: "Very Low", 1: "Low", 2: "Moderate", 3: "High", 4: "Very High"}
    st.success(f"Predicted Stress Level: {stress_labels[prediction[0]]}")
