#!/usr/bin/env python3
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_model():
    # Check if we have enough data
    csv_file = 'ai-engine/data/build_metrics.csv'
    
    if not os.path.exists(csv_file):
        print("⚠️ No metrics data found. Need more build history.")
        return False
    
    data = pd.read_csv(csv_file)
    
    # Need at least 10 builds to train a decent model
    if len(data) < 10:
        print(f"⚠️ Insufficient data: {len(data)} builds. Need at least 10.")
        return False
    
    # Prepare features (timestamp, duration) and target (status)
    data['timestamp'] = pd.to_numeric(data['timestamp'], errors='coerce')
    data['duration_seconds'] = data['duration'].apply(convert_duration)
    
    # Remove rows with invalid data
    data = data.dropna()
    
    if len(data) < 10:
        print("⚠️ Not enough valid data after cleaning.")
        return False
    
    # Features and target
    X = data[['timestamp', 'duration_seconds']]
    y = data['status'].apply(lambda x: 1 if x == 'SUCCESS' else 0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    os.makedirs('ai-engine/model', exist_ok=True)
    joblib.dump(model, 'ai-engine/model/build_predictor.pkl')
    
    accuracy = model.score(X_test, y_test)
    print(f"✅ Model trained successfully! Accuracy: {accuracy:.2f}")
    return True

def convert_duration(duration_str):
    """Convert duration string to seconds"""
    try:
        if 'min' in duration_str:
            minutes = float(duration_str.split(' min')[0])
            return minutes * 60
        elif 'sec' in duration_str:
            seconds = float(duration_str.split(' sec')[0])
            return seconds
        return float(duration_str)
    except:
        return 0

if __name__ == "__main__":
    train_model()
