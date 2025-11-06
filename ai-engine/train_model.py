#!/usr/bin/env python3
import pandas as pd
import os
import sys
import csv
from sklearn.ensemble import RandomForestClassifier
import joblib

def convert_duration_to_seconds(duration_str):
    """Convert Jenkins duration string to seconds"""
    try:
        duration_str = str(duration_str).lower().strip()
        print(f"🕒 Parsing duration: {duration_str}")
        
        # Handle "and counting" format
        if 'and counting' in duration_str:
            time_part = duration_str.split(' and')[0]
            duration_str = time_part
        
        # Handle different duration formats
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

def read_metrics_manual(csv_file):
    """Manually read and parse the CSV file to avoid pandas issues"""
    builds = []
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
            
            if len(lines) < 2:
                print("❌ Not enough data in CSV file")
                return []
            
            # Extract header and data rows
            header = lines[0]
            data_rows = lines[1:]
            
            print(f"📊 CSV Header: {header}")
            print(f"📊 Data rows: {len(data_rows)}")
            
            for i, row in enumerate(data_rows):
                if len(row) >= 3:
                    build_data = {
                        'timestamp': row[0],
                        'duration': row[1], 
                        'status': row[2]
                    }
                    builds.append(build_data)
                    print(f"  Build {i+1}: {build_data}")
                else:
                    print(f"⚠️ Skipping invalid row {i+1}: {row}")
        
        return builds
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return []

def train_model():
    print("🤖 Starting AI model training...")
    
    csv_file = 'ai-engine/data/build_metrics.csv'
    
    if not os.path.exists(csv_file):
        print("❌ No metrics data file found.")
        return False
    
    # Read data manually to avoid pandas issues
    builds = read_metrics_manual(csv_file)
    
    if len(builds) < 1:
        print(f"⚠️ Insufficient data: {len(builds)} builds. Need at least 1.")
        return False
    
    print(f"✅ Found {len(builds)} builds! Processing...")
    
    try:
        # Convert to DataFrame
        data = pd.DataFrame(builds)
        print(f"📈 Created DataFrame with {len(data)} rows")
        print(f"🔍 DataFrame columns: {list(data.columns)}")
        
        # Convert duration to seconds
        data['duration_seconds'] = data['duration'].apply(convert_duration_to_seconds)
        
        # Convert timestamp to numerical
        data['timestamp_numeric'] = pd.to_numeric(data['timestamp'], errors='coerce')
        
        # Convert status to numerical (1 for SUCCESS, 0 for FAILURE)
        data['status_numeric'] = data['status'].apply(lambda x: 1 if str(x).strip() == 'SUCCESS' else 0)
        
        # Remove invalid rows
        data_clean = data.dropna()
        
        print(f"📊 Cleaned data: {len(data_clean)} valid records")
        
        if len(data_clean) < 1:
            print("❌ No valid data after cleaning")
            return False
        
        # Prepare features and target
        X = data_clean[['timestamp_numeric', 'duration_seconds']]
        y = data_clean['status_numeric']
        
        print(f"🎯 Training on {len(X)} samples")
        
        # Train model - use all data for training when we have few samples
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        # For single build, we can't calculate proper accuracy
        if len(X) == 1:
            print("✅ Model trained with single build! (Will improve with more data)")
            accuracy = 1.0  # Perfect on training data
        else:
            accuracy = model.score(X, y)  # Training accuracy
            print(f"✅ Model trained! Accuracy: {accuracy:.2f}")
        
        # Save model
        os.makedirs('ai-engine/model', exist_ok=True)
        model_path = 'ai-engine/model/build_predictor.pkl'
        joblib.dump(model, model_path)
        
        print(f"💾 Model saved to: {model_path}")
        print("🎉 AI is now active! Will make real predictions.")
        return True
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)
