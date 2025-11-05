#!/usr/bin/env python3
import joblib
import json
import os
import pandas as pd
from datetime import datetime

def predict_build():
    print("🧠 Starting build prediction...")
    
    model_path = 'ai-engine/model/build_predictor.pkl'
    csv_file = 'ai-engine/data/build_metrics.csv'
    
    # Check if model exists and is valid
    if not os.path.exists(model_path):
        print("❌ No trained model found.")
        return create_default_prediction()
    
    try:
        # Load model
        model = joblib.load(model_path)
        print("✅ Loaded trained model")
        
        # Load recent data to get patterns
        if os.path.exists(csv_file):
            data = pd.read_csv(csv_file)
            if len(data) > 0:
                # Use the average duration from recent builds
                recent_durations = data['duration'].tail(5).apply(convert_duration_to_seconds)
                avg_duration = recent_durations.mean()
            else:
                avg_duration = 60
        else:
            avg_duration = 60
        
        # Create features for current build
        current_timestamp = int(datetime.now().timestamp())
        features = [[current_timestamp, avg_duration]]
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        predicted_status = "SUCCESS" if prediction == 1 else "FAILURE"
        confidence = max(probability)
        
        result = {
            "predicted_status": predicted_status,
            "confidence": round(confidence, 2),
            "predicted_duration": f"{int(avg_duration)} sec",
            "model_used": True,
            "builds_trained_on": len(data) if os.path.exists(csv_file) else 0
        }
        
        print(f"✅ Prediction: {predicted_status} (confidence: {confidence:.2f})")
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return create_default_prediction()
    
    # Save results
    os.makedirs('ai-engine/results', exist_ok=True)
    with open('ai-engine/results/predictions.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

def create_default_prediction():
    """Create a default prediction when no model is available"""
    result = {
        "predicted_status": "SUCCESS",
        "confidence": 0.5,
        "predicted_duration": "60 sec", 
        "model_used": False,
        "note": "Using default prediction - model not yet trained"
    }
    
    os.makedirs('ai-engine/results', exist_ok=True)
    with open('ai-engine/results/predictions.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("📊 Using default prediction (model not trained)")
    return result

def convert_duration_to_seconds(duration_str):
    """Helper function to convert duration"""
    try:
        duration_str = str(duration_str).lower()
        if 'min' in duration_str:
            return float(''.join(filter(str.isdigit, duration_str))) * 60
        elif 'sec' in duration_str:
            return float(''.join(filter(str.isdigit, duration_str)))
        else:
            return float(duration_str)
    except:
        return 60.0

if __name__ == "__main__":
    predict_build()
