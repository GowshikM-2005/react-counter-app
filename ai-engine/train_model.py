import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

DATA_FILE = "ai-engine/data/build_metrics.csv"
MODEL_FILE = "ai-engine/model.pkl"

# 🧩 1. Check file existence
if not os.path.exists(DATA_FILE):
    print(f"⚠️ Data file not found: {DATA_FILE}")
    exit(1)

# 🧩 2. Read CSV safely
try:
    df = pd.read_csv(DATA_FILE)
except Exception as e:
    print(f"⚠️ Error reading {DATA_FILE}: {e}")
    exit(1)

# 🧩 3. Auto-assign headers if missing
if "duration" not in df.columns or "status" not in df.columns:
    print("⚙️ CSV has no headers, assigning default column names...")
    df.columns = ["timestamp", "duration", "status"]

# 🧩 4. Clean up duration column (remove ' sec', handle missing)
df['duration'] = df['duration'].astype(str).str.replace(r'[^0-9.]', '', regex=True)
df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0)

# 🧩 5. Encode status values
df['status'] = df['status'].map({'SUCCESS': 0, 'FAILURE': 1})
df = df.dropna(subset=['status'])

if df.empty:
    print("⚠️ No valid rows to train on (empty dataset).")
    exit(1)

# 🧩 6. Train-test split
X = df[['duration']]
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧩 7. Train simple model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# 🧩 8. Save model
os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
joblib.dump(model, MODEL_FILE)

print(f"✅ Model trained successfully! Saved to {MODEL_FILE}")
print(f"📊 Training rows: {len(X_train)}, Testing rows: {len(X_test)}")
