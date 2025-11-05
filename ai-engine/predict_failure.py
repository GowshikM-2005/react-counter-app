#!/usr/bin/env python3
import joblib
import json
import os
import pandas as pd
from datetime import datetime

def predict_build():
    model_path = 'ai-engine/model/build_predictor.pkl'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("⚠️ No trained model found.")
        return create_default_prediction()
    
    try:
        # Load model
        model = joblib.load(model_path)
        
        # Create features for current build
        current_timestamp = int(datetime.now().timestamp())
        # Use average duration as placeholder (you can improve this)
        avg_duration = 60  # seconds
        
        features = [[current_timestamp, avg_duration]]
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        predicted_status = "SUCCESS" if prediction == 1 else "FAILURE"
        confidence = max(probability)
        
        result = {
            "predicted_status": predicted_status,
            "confidence": round(confidence, 2),
            "predicted_duration": "60 sec",
            "model_used": True
        }
        
    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return create_default_prediction()
    
    # Save results
    os.makedirs('ai-engine/results', exist_ok=True)
    with open('ai-engine/results/predictions.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✅ Prediction: {result}")
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
    
    return result

if __name__ == "__main__":
    predict_build()
