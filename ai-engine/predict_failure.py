#!/usr/bin/env python3
import joblib
import json
import os
import pandas as pd
import csv
from datetime import datetime

def convert_duration_to_seconds(duration_str):
    """Convert Jenkins duration string to seconds"""
    try:
        duration_str = str(duration_str).lower().strip()
        
        if 'and counting' in duration_str:
            time_part = duration_str.split(' and')[0]
            duration_str = time_part
        
        if 'hr' in duration_str and 'min' in duration_str:
            parts = duration_str.split()
            hours = int(parts[0]) 
            minutes = int(parts[2])
            return hours * 3600 + minutes * 60
            
        elif 'min' in duration_str and 'sec' in duration_str:
            parts = duration_str.split()
            minutes = int(parts[0])
            seconds = int(parts[2])
            return minutes * 60 + seconds
            
        elif 'min' in duration_str:
            minutes = int(''.join(filter(str.isdigit, duration_str)))
            return minutes * 60
            
        elif 'sec' in duration_str:
            seconds = int(''.join(filter(str.isdigit, duration_str)))
            return seconds
            
        else:
            return float(duration_str)
            
    except Exception as e:
        print(f"⚠️ Could not parse duration: {duration_str}, error: {e}")
        return 60.0

def get_build_count_manual(csv_file):
    """Manually count builds"""
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            return len(rows) - 1 if len(rows) > 1 else 0
    except:
        return 0

def predict_build():
    print("🧠 Starting build prediction...")
    
    model_path = 'ai-engine/model/build_predictor.pkl'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("❌ No trained model found.")
        return create_default_prediction()
    
    try:
        # Load model
        model = joblib.load(model_path)
        print("✅ Loaded trained model")
        
        # Get build count for info
        csv_file = 'ai-engine/data/build_metrics.csv'
        build_count = get_build_count_manual(csv_file)
        
        # Use average duration (we'll improve this later)
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
            "builds_trained_on": build_count
        }
        
        print(f"✅ AI Prediction: {predicted_status} (confidence: {confidence:.2f})")
        
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

if __name__ == "__main__":
    predict_build()
