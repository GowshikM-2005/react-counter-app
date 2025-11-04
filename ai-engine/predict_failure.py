import joblib
import json
import sys
import random
from datetime import datetime

MODEL_PATH = "ai-engine/model.pkl"
RESULT_FILE = "ai-engine/results/predictions.json"

try:
    model = joblib.load(MODEL_PATH)
except:
    print("⚠️ No model found. Defaulting to SUCCESS prediction.")
    sys.exit(0)

# Simulated current pipeline data
build_duration = random.uniform(50, 300)  # You can replace with actual metrics
prediction = model.predict([[build_duration]])[0]

result = {
    "timestamp": datetime.now().isoformat(),
    "predicted_status": "SUCCESS" if prediction == 1 else "FAILURE",
    "predicted_duration": build_duration
}

print("🧠 Prediction Result:", result)

# Save result for analysis
with open(RESULT_FILE, "w") as f:
    json.dump(result, f)
