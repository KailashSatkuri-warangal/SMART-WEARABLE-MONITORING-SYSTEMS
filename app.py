import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Load model and scaler
try:
    rf_model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or Scaler file not found.")
    rf_model, scaler = None, None

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('health_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS health_data
                 (id INTEGER, timestamp TEXT, steps REAL, distance REAL, calories REAL, weight REAL, bmi REAL,
                  heart_rate REAL, mobile_usage REAL, water_intake REAL, systolic_bp REAL, diastolic_bp REAL)''')
    conn.commit()
    conn.close()

init_db()

# Evaluate Model
def evaluate_model():
    df = pd.read_csv('merged_fitbit_dataset.csv', low_memory=False)
    features = ['TotalSteps', 'TotalDistance', 'VeryActiveMinutes',
                'FairlyActiveMinutes', 'LightlyActiveMinutes',
                'SedentaryMinutes', 'WeightKg', 'BMI']
    target = 'Calories'
    X = df[features].fillna(df[features].median())
    y = df[target]
    X_scaled = scaler.transform(X)
    y_pred = rf_model.predict(X_scaled)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    fig, ax = plt.subplots()
    ax.scatter(y[:1000], y_pred[:1000], alpha=0.5)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_xlabel("Actual Calories")
    ax.set_ylabel("Predicted Calories")
    ax.set_title(f"MSE: {mse:.2f}, R²: {r2:.2f}")
    st.pyplot(fig)
    return mse, r2

# Health Recommendations
def get_recommendations(data):
    recommendations = []
    if data['steps'] < 5000:
        recommendations.append("Increase daily steps to at least 10,000 for better health.")
    if data['bmi'] > 24.9:
        recommendations.append("Consider a balanced diet to manage BMI.")
    if data['calories'] < 2000:
        recommendations.append("Ensure adequate calorie intake for energy.")
    if data['heart_rate'] > 100:
        recommendations.append("Monitor heart rate; consult a doctor if consistently high.")
    if data['mobile_usage'] > 50:
        recommendations.append("Reduce screen time and consider stress-relief exercises.")
    if data['water_intake'] < 1.5:
        recommendations.append("Aim to drink at least 1.5-2 liters of water daily.")
    if data['systolic_bp'] > 130 or data['diastolic_bp'] > 90:
        recommendations.append("Check blood pressure regularly; consult a healthcare provider if elevated.")
    return recommendations

# Prescription Static Reader
def display_prescription():
    try:
        with open("precription.txt", "r") as file:
            content = file.read()
            st.subheader("Doctor's Prescription")
            st.text(content)
    except FileNotFoundError:
        st.warning("Prescription file not found.")

# Streamlit Interface
st.title("Health Data Analysis & Calorie Prediction")

st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose Page", ["Evaluate Model", "Enter Health Data", "View Prescription"])

if page == "Evaluate Model":
    st.header("Model Evaluation")
    if rf_model and scaler:
        mse, r2 = evaluate_model()
        st.success(f"MSE: {mse:.2f}, R² Score: {r2:.2f}")
    else:
        st.warning("Model not available. Please upload rf_model.pkl and scaler.pkl")

elif page == "Enter Health Data":
    st.header("Health Data Input")
    with st.form("health_form"):
        id = st.number_input("User ID", min_value=1)
        steps = st.number_input("Total Steps", min_value=0)
        distance = st.number_input("Distance (km)", min_value=0.0)
        weight = st.number_input("Weight (kg)", min_value=0.0)
        bmi = st.number_input("BMI", min_value=0.0)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=0)
        mobile_usage = st.number_input("Mobile Usage (hours)", min_value=0.0)
        water_intake = st.number_input("Water Intake (liters)", min_value=0.0)
        systolic_bp = st.number_input("Systolic BP", min_value=0)
        diastolic_bp = st.number_input("Diastolic BP", min_value=0)
        submit = st.form_submit_button("Submit")

    if submit:
        if not rf_model or not scaler:
            st.error("Model or Scaler not loaded.")
        else:
            df_input = pd.DataFrame([{
                'TotalSteps': steps, 'TotalDistance': distance,
                'VeryActiveMinutes': 0, 'FairlyActiveMinutes': 0,
                'LightlyActiveMinutes': 0, 'SedentaryMinutes': 0,
                'WeightKg': weight, 'BMI': bmi
            }])
            df_input.fillna(df_input.median(), inplace=True)
            scaled_input = scaler.transform(df_input)
            calories = rf_model.predict(scaled_input)[0]
            st.success(f"Predicted Calories: {calories:.2f}")

            conn = sqlite3.connect('health_data.db')
            c = conn.cursor()
            c.execute("INSERT INTO health_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (id, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), steps, distance, calories,
                       weight, bmi, heart_rate, mobile_usage, water_intake, systolic_bp, diastolic_bp))
            conn.commit()
            conn.close()

            data_dict = {
                'steps': steps, 'calories': calories, 'bmi': bmi,
                'heart_rate': heart_rate, 'mobile_usage': mobile_usage,
                'water_intake': water_intake, 'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp
            }

            st.subheader("Health Recommendations")
            for rec in get_recommendations(data_dict):
                st.write("- " + rec)

elif page == "View Prescription":
    display_prescription()
