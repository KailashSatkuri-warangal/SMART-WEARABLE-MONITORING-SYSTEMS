import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def evaluate_model():
    df = pd.read_csv('merged_fitbit_dataset.csv', low_memory=False)

    features = ['TotalSteps', 'TotalDistance', 'VeryActiveMinutes',
                'FairlyActiveMinutes', 'LightlyActiveMinutes',
                'SedentaryMinutes', 'WeightKg', 'BMI']
    target = 'Calories'  # updated to match your dataset

    X = df[features].fillna(df[features].median())
    y = df[target]

    scaler = joblib.load('scaler.pkl')
    rf_model = joblib.load('rf_model.pkl')

    X_scaled = scaler.transform(X)
    y_pred = rf_model.predict(X_scaled)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, color='blue', alpha=0.5, label='Predicted vs Actual')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Ideal Fit')
    plt.xlabel('Actual Calories')
    plt.ylabel('Predicted Calories')
    plt.title(f'Actual vs Predicted Calories (MSE: {mse:.2f}, R²: {r2:.2f})')
    plt.legend()
    plt.grid(True)
    plt.savefig('actual_vs_predicted.png')
    plt.show()

evaluate_model()
