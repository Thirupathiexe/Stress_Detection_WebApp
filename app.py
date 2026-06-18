import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("XGB_model.pkl")

# Set up the app title and description
st.title("Stress Detection Using Machine Learning Approach for Mental Health Monitoring ")
st.markdown(
    """
    Welcome! This app predicts your stress level based on physiological parameters.
    Enter your details below and let's see how you're doing! 😊
    """
)

# User inputs with sliders and number fields
snoring_range = st.slider("😴 Snoring Range", 45, 100, 71)
respiration_rate = st.slider("🌬️ Respiration Rate", 16, 49, 22)
body_temperature = st.number_input("🌡️ Body Temperature (°F)", min_value=85.0, max_value=166.2, value=93.5)
limb_movement = st.slider("💃 Limb Movement", 4, 47, 12)
blood_oxygen = st.slider("🩸 Blood Oxygen Level (%)", 82, 154, 91)
eye_movement = st.number_input("👀 Eye Movement", 60, 185, 89)
hours_of_sleep = st.slider("🛌 Hours of Sleep", 0.0, 20.2, 3.8)
heart_rate = st.number_input("❤️ Heart Rate", min_value=50.0, max_value=158.7, value=65.0)

# Predict button
if st.button("Predict Stress Level 🔮"):
    # Create a NumPy array with input values
    input_data = np.array([
        snoring_range, respiration_rate, body_temperature, limb_movement,
        blood_oxygen, eye_movement, hours_of_sleep, heart_rate
    ]).reshape(1, -1)
    
    # Predict the stress level
    prediction = model.predict(input_data)[0]
    
    # Define motivational messages
    motivation = {
        0: "🎉 Hooray! You're stress-free! Keep up the positive vibes and enjoy life! 😊",
        1: "🙂 You're doing well! A little relaxation can help maintain this balance. Keep going!",
        2: "😌 Mild stress detected. Take some deep breaths and enjoy a break! You've got this!",
        3: "😟 Your stress level is high! Try meditation, a short walk, or talking to a friend. Stay strong! 💪",
        4: "🚨 Very high stress! Please prioritize your health, rest, and seek support if needed. You matter! ❤️"
    }
    
    # Display the prediction and motivation
    st.subheader(f"Your Predicted Stress Level: {prediction}")
    st.success(motivation[prediction])
