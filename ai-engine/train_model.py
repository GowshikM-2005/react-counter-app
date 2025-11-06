#!/usr/bin/env python3
import pandas as pd
import os
import sys
import csv
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

def count_builds_manual(csv_file):
    """Manually count builds in CSV to avoid pandas issues"""
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            # Count non-header rows
            build_count = len(rows) - 1 if len(rows) > 0 else 0
            print(f"📊 Manual count: {build_count} builds (from {len(rows)} total rows)")
            return build_count, rows
    except Exception as e:
        print(f"❌ Error reading CSV manually: {e}")
        return 0, []

def train_model():
    print("🤖 Starting AI model training...")
    
    # Check if we have enough data
    csv_file = 'ai-engine/data/build_metrics.csv'
    
    if not os.path.exists(csv_file):
        print("❌ No metrics data file found.")
        return False
    
    try:
        # First, manually count the builds to avoid pandas issues
        build_count, csv_rows = count_builds_manual(csv_file)
        
        if build_count < 1:
            print(f"⚠️ Insufficient data: {build_count} builds. Need at least 1.")
            return False
        
        print(f"✅ Found {build_count} builds! Processing...")
        
        # Try to read with pandas, but if it fails, create DataFrame manually
        try:
            data = pd.read_csv(csv_file)
            print(f"📈 Pandas successfully read {len(data)} rows")
        except Exception as e:
            print(f"⚠️ Pandas read failed, creating manual DataFrame: {e}")
            # Create DataFrame manually from CSV rows
            if len(csv_rows) > 1:
                headers = csv_rows[0]
                data_rows = csv_rows[1:]
                data = pd.DataFrame(data_rows, columns=headers)
                print(f"📈 Manual DataFrame created with {len(data)} rows")
            else:
                print("❌ Not enough rows to create DataFrame")
                return False
        
        # Ensure we have the expected columns
        print(f"🔍 DataFrame columns: {list(data.columns)}")
        print(f"🔍 DataFrame content:\n{data}")
        
        # Convert duration to seconds
        data['duration_seconds'] = data['duration'].apply(convert_duration_to_seconds)
        
        # Convert timestamp to numerical feature
        data['timestamp_numeric'] = pd.to_numeric(data['timestamp'], errors='coerce')
        
        # Convert status to numerical (1 for SUCCESS, 0 for FAILURE)
        data['status_numeric'] = data['status'].apply(lambda x: 1 if str(x).strip() == 'SUCCESS' else 0)
        
        # Remove any rows with invalid data
        data_clean = data.dropna()
        
        print(f"📈 Cleaned data: {len(data_clean)} valid records")
        
        if len(data_clean) < 1:
            print(f"⚠️ No valid data after cleaning")
            return False
        
        # Prepare features and target
        X = data_clean[['timestamp_numeric', 'duration_seconds']]
        y = data_clean['status_numeric']
        
        print(f"🎯 Features shape: {X.shape}")
        print(f"🎯 Target shape: {y.shape}")
        
        # Train model with all available data
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # For single build, we can't calculate accuracy but we can still train
        if len(X) > 1:
            accuracy = model.score(X, y)  # This is training accuracy, not ideal but works
            print(f"✅ Model trained! Accuracy: {accuracy:.2f}")
        else:
            print(f"✅ Model trained with single build! (Will improve with more data)")
        
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
