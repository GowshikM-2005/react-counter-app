import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

DATA_FILE = "ai-engine/data/build_metrics.csv"
MODEL_FILE = "ai-engine/model.pkl"

if not os.path.exists(DATA_FILE):
    print("No training data found, skipping model training.")
    exit(0)

df = pd.read_csv(DATA_FILE)
df['duration'] = df['duration'].str.replace('s','').astype(float)
df['label'] = df['status'].apply(lambda x: 1 if x == 'SUCCESS' else 0)

X = df[['duration']]
y = df['label']

model = RandomForestClassifier(n_estimators=50)
model.fit(X, y)
joblib.dump(model, MODEL_FILE)

print(f"✅ Model trained and saved at {MODEL_FILE}")
