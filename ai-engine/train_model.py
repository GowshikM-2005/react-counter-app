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
        return 60.0

def read_metrics_manual(csv_file):
    """Manually read and parse the CSV file to avoid pandas issues"""
    builds = []
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
            
            if len(lines) < 2:
                print("❌ Not enough data in CSV file (need at least header + 1 data row)")
                return []
            
            # Extract header and data rows
            header = lines[0]
            data_rows = lines[1:]
            
            print(f"📊 CSV Header: {header}")
            print(f"📊 Data rows found: {len(data_rows)}")
            
            # Validate header
            expected_headers = ['timestamp', 'duration', 'status']
            if header != expected_headers:
                print(f"⚠️ Header mismatch. Expected: {expected_headers}, Got: {header}")
                print("🔧 Attempting to fix header issues...")
            
            for i, row in enumerate(data_rows):
                if len(row) >= 3:
                    build_data = {
                        'timestamp': row[0].strip(),
                        'duration': row[1].strip(), 
                        'status': row[2].strip()
                    }
                    builds.append(build_data)
                    print(f"  Build {i+1}: timestamp={build_data['timestamp']}, duration={build_data['duration']}, status={build_data['status']}")
                else:
                    print(f"⚠️ Skipping invalid row {i+1} (not enough columns): {row}")
        
        print(f"✅ Successfully loaded {len(builds)} builds from CSV")
        return builds
        
    except Exception as e:
        print(f"❌ Error reading CSV file {csv_file}: {e}")
        return []

def count_unique_builds(builds):
    """Count unique builds based on timestamp to avoid duplicates"""
    unique_timestamps = set()
    unique_builds = []
    
    for build in builds:
        timestamp = build['timestamp']
        if timestamp not in unique_timestamps:
            unique_timestamps.add(timestamp)
            unique_builds.append(build)
    
    print(f"📈 Unique builds: {len(unique_builds)} (removed {len(builds) - len(unique_builds)} duplicates)")
    return unique_builds

def train_model():
    print("🤖 Starting AI model training...")
    print("=" * 50)
    
    csv_file = 'ai-engine/data/build_metrics.csv'
    
    if not os.path.exists(csv_file):
        print("❌ No metrics data file found.")
        print("💡 Make sure the 'Restore AI Data' stage is working in your pipeline")
        return False
    
    # Read data manually to avoid pandas issues
    builds = read_metrics_manual(csv_file)
    
    if len(builds) < 1:
        print(f"⚠️ Insufficient data: {len(builds)} builds. Need at least 1.")
        return False
    
    # Remove duplicate builds (in case of data restoration issues)
    unique_builds = count_unique_builds(builds)
    
    print(f"🎯 Training AI model with {len(unique_builds)} unique builds...")
    
    try:
        # Convert to DataFrame
        data = pd.DataFrame(unique_builds)
        print(f"📈 Created DataFrame with {len(data)} rows")
        print(f"🔍 DataFrame columns: {list(data.columns)}")
        
        # Show sample of data
        print("📋 Data sample:")
        for i, row in data.iterrows():
            print(f"  Row {i}: {row['timestamp']} | {row['duration']} | {row['status']}")
        
        # Convert duration to seconds
        data['duration_seconds'] = data['duration'].apply(convert_duration_to_seconds)
        
        # Convert timestamp to numerical
        data['timestamp_numeric'] = pd.to_numeric(data['timestamp'], errors='coerce')
        
        # Convert status to numerical (1 for SUCCESS, 0 for FAILURE)
        data['status_numeric'] = data['status'].apply(lambda x: 1 if str(x).strip().upper() == 'SUCCESS' else 0)
        
        # Remove invalid rows
        initial_count = len(data)
        data_clean = data.dropna()
        removed_count = initial_count - len(data_clean)
        
        if removed_count > 0:
            print(f"⚠️ Removed {removed_count} rows with invalid data")
        
        print(f"📊 Cleaned data: {len(data_clean)} valid records")
        
        if len(data_clean) < 1:
            print("❌ No valid data after cleaning")
            return False
        
        # Prepare features and target
        X = data_clean[['timestamp_numeric', 'duration_seconds']]
        y = data_clean['status_numeric']
        
        print(f"🎯 Training on {len(X)} samples")
        print(f"   - Successes: {sum(y == 1)}")
        print(f"   - Failures: {sum(y == 0)}")
        
        # Train model - use all data for training when we have few samples
        model = RandomForestClassifier(
            n_estimators=min(50, len(X) * 10),  # Adjust based on data size
            random_state=42,
            max_depth=min(10, len(X))  # Prevent overfitting on small data
        )
        model.fit(X, y)
        
        # Calculate training accuracy
        training_accuracy = model.score(X, y)
        
        if len(X) == 1:
            print("✅ Model trained with single build! (Will improve with more data)")
            print(f"   Training accuracy: {training_accuracy:.2f}")
        else:
            print(f"✅ Model trained! Training accuracy: {training_accuracy:.2f}")
        
        # Save model
        os.makedirs('ai-engine/model', exist_ok=True)
        model_path = 'ai-engine/model/build_predictor.pkl'
        joblib.dump(model, model_path)
        
        print(f"💾 Model saved to: {model_path}")
        print(f"🎉 AI is now active! Model trained on {len(unique_builds)} builds.")
        print("=" * 50)
        
        # Return the number of builds used for training
        return len(unique_builds)
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = train_model()
    if result is False:
        print("❌ AI training failed")
        sys.exit(1)
    else:
        print(f"✅ AI training completed successfully with {result} builds")
        sys.exit(0)
