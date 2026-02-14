# Quick test script: test_multiclass_flag.py
from model_xgb import train_model
from data_loader import fetch_historical_data

df = fetch_historical_data("NVDA", period="1y", interval="1d")
artifact = train_model(df, symbol="NVDA", mode="daily")

print(f"Model type: {artifact.get('target_type')}")
print(f"Num classes: {artifact.get('num_classes')}")

# Should print:
# Model type: multiclass
# Num classes: 5
