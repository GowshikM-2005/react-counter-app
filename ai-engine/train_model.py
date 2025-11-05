#!/usr/bin/env python3
import pandas as pd
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_model():
    print("🤖 Starting AI model training...")
    
    # Check if we have enough data
    csv_file = 'ai-engine/data/build_metrics.csv'
    
    if not os.path.exists(csv_file):
        print("❌ No metrics data file found.")
        return False
    
    try:
        data = pd.read_csv(csv_file)
        print(f"📊 Found {len(data)} build records in CSV")
        
        # Check if we have the header + at least 10 builds
        if len(data) < 10:
            print(f"⚠️ Insufficient data: {len(data)} builds. Need at least 10.")
            return False
        
        print("✅ Enough data available! Processing...")
        
        # Convert duration to seconds
        data['duration_seconds'] = data['duration'].apply(convert_duration_to_seconds)
        
        # Convert timestamp to numerical feature
        data['timestamp_numeric'] = pd.to_numeric(data['timestamp'], errors='coerce')
        
        # Convert status to numerical (1 for SUCCESS, 0 for FAILURE)
        data['status_numeric'] = data['status'].apply(lambda x: 1 if x.strip() == 'SUCCESS' else 0)
        
        # Remove any rows with invalid data
        data_clean = data.dropna()
        
        print(f"📈 Cleaned data: {len(data_clean)} valid records")
        print(f"   Successes: {len(data_clean[data_clean['status_numeric'] == 1])}")
        print(f"   Failures: {len(data_clean[data_clean['status_numeric'] == 0])}")
        
        if len(data_clean) < 10:
            print(f"⚠️ Not enough valid data after cleaning: {len(data_clean)} records")
            return False
        
        # Prepare features and target
        X = data_clean[['timestamp_numeric', 'duration_seconds']]
        y = data_clean['status_numeric']
        
        print(f"🎯 Features shape: {X.shape}")
        print(f"🎯 Target shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        accuracy = model.score(X_test, y_test)
        print(f"✅ Model trained! Accuracy: {accuracy:.2f}")
        
        # Save model
        os.makedirs('ai-engine/model', exist_ok=True)
        model_path = 'ai-engine/model/build_predictor.pkl'
        joblib.dump(model, model_path)
        
        print(f"💾 Model saved to: {model_path}")
        return True
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_duration_to_seconds(duration_str):
    """Convert Jenkins duration string to seconds"""
    try:
        duration_str = str(duration_str).lower().strip()
        
        # Handle different duration formats
        if 'hr' in duration_str and 'min' in duration_str:
            # Format: "1 hr 20 min"
            parts = duration_str.split()
            hours = int(parts[0]) if 'hr' in parts[0] else 0
            minutes = int(parts[2]) if 'min' in parts[2] else 0
            return hours * 3600 + minutes * 60
            
        elif 'min' in duration_str:
            # Format: "20 min" or "20 min 30 sec"
            if 'sec' in duration_str:
                parts = duration_str.split()
                minutes = int(parts[0]) 
                seconds = int(parts[2])
                return minutes * 60 + seconds
            else:
                minutes = float(''.join(filter(str.isdigit, duration_str)))
                return minutes * 60
                
        elif 'sec' in duration_str:
            # Format: "30 sec"
            seconds = float(''.join(filter(str.isdigit, duration_str)))
            return seconds
            
        elif 'ms' in duration_str:
            # Format: "1500 ms"
            ms = float(''.join(filter(str.isdigit, duration_str)))
            return ms / 1000.0
            
        else:
            # Try to parse as plain number (assuming seconds)
            return float(duration_str)
            
    except Exception as e:
        print(f"⚠️ Could not parse duration: {duration_str}, error: {e}")
        return 60.0  # Default to 60 seconds

if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)
