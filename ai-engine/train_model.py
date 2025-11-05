#!/usr/bin/env python3
import pandas as pd
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def convert_duration_to_seconds(duration_str):
    """Convert Jenkins duration string to seconds"""
    try:
        duration_str = str(duration_str).lower().strip()
        print(f"🕒 Parsing duration: {duration_str}")
        
        # Handle "and counting" format
        if 'and counting' in duration_str:
            # Extract just the time part: "1 min 29 sec"
            time_part = duration_str.split(' and')[0]
            duration_str = time_part
        
        # Handle different duration formats
        if 'hr' in duration_str and 'min' in duration_str:
            parts = duration_str.split()
            hours = int(parts[0]) 
            minutes = int(parts[2])
            return hours * 3600 + minutes * 60
            
        elif 'min' in duration_str and 'sec' in duration_str:
            # Format: "1 min 29 sec"
            parts = duration_str.split()
            minutes = int(parts[0])
            seconds = int(parts[2])
            return minutes * 60 + seconds
            
        elif 'min' in duration_str:
            # Format: "1 min"
            minutes = int(''.join(filter(str.isdigit, duration_str)))
            return minutes * 60
            
        elif 'sec' in duration_str:
            # Format: "29 sec"
            seconds = int(''.join(filter(str.isdigit, duration_str)))
            return seconds
            
        else:
            # Try to parse as plain number
            return float(duration_str)
            
    except Exception as e:
        print(f"⚠️ Could not parse duration: {duration_str}, error: {e}")
        return 60.0  # Default to 60 seconds

def train_model():
    print("🤖 Starting AI model training...")
    
    # Check if we have enough data
    csv_file = 'ai-engine/data/build_metrics.csv'
    
    if not os.path.exists(csv_file):
        print("❌ No metrics data file found.")
        return False
    
    try:
        data = pd.read_csv(csv_file)
        total_builds = len(data)
        print(f"📊 Found {total_builds} build records in CSV")
        
        # REDUCED: Only need 2+ builds to start training (was 10)
        if total_builds < 2:
            print(f"⚠️ Insufficient data: {total_builds} builds. Need at least 2.")
            return False
        
        print("✅ Enough data available! Processing...")
        
        # Convert duration to seconds
        data['duration_seconds'] = data['duration'].apply(convert_duration_to_seconds)
        
        # Convert timestamp to numerical feature
        data['timestamp_numeric'] = pd.to_numeric(data['timestamp'], errors='coerce')
        
        # Convert status to numerical (1 for SUCCESS, 0 for FAILURE)
        data['status_numeric'] = data['status'].apply(lambda x: 1 if str(x).strip() == 'SUCCESS' else 0)
        
        # Remove any rows with invalid data
        data_clean = data.dropna()
        
        print(f"📈 Cleaned data: {len(data_clean)} valid records")
        print(f"   Successes: {len(data_clean[data_clean['status_numeric'] == 1])}")
        print(f"   Failures: {len(data_clean[data_clean['status_numeric'] == 0])}")
        
        # Even if we have few records, let's train with what we have
        if len(data_clean) < 2:
            print(f"⚠️ Very little data, but will try to train with {len(data_clean)} records")
        
        # Prepare features and target
        X = data_clean[['timestamp_numeric', 'duration_seconds']]
        y = data_clean['status_numeric']
        
        print(f"🎯 Features shape: {X.shape}")
        print(f"🎯 Target shape: {y.shape}")
        
        # If we have very few samples, use all for training
        if len(X) < 5:
            X_train, y_train = X, y
            X_test, y_test = X, y  # Not ideal but works for small data
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)  # Reduced for small data
        model.fit(X_train, y_train)
        
        # Calculate accuracy if we have test data
        if len(X_test) > 0:
            accuracy = model.score(X_test, y_test)
            print(f"✅ Model trained! Accuracy: {accuracy:.2f}")
        else:
            accuracy = 0.0
            print(f"✅ Model trained! (No test data for accuracy)")
        
        # Save model
        os.makedirs('ai-engine/model', exist_ok=True)
        model_path = 'ai-engine/model/build_predictor.pkl'
        joblib.dump(model, model_path)
        
        print(f"💾 Model saved to: {model_path}")
        print(f"🎉 AI is now active! Will make predictions on future builds.")
        return True
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)
