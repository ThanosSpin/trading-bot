from data_loader import fetch_latest_price
from model_xgb import compute_signals
from strategy import compute_strategy_decisions

# Get current PLTR signal
predictions = {}
diagnostics = {}

sig = compute_signals("PLTR", lookback_minutes=2400, intraday_weight=0.60)
predictions["PLTR"] = sig.get("final_prob")
diagnostics["PLTR"] = sig

# Get strategy decision
decisions = compute_strategy_decisions(predictions, symbols=["PLTR"], diagnostics=diagnostics)

print(f"\n📊 PLTR Decision Preview")
print(f"Current Price: ${sig.get('price'):.2f}")
print(f"Action: {decisions['PLTR']['action'].upper()}")
print(f"Quantity: {decisions['PLTR']['qty']}")
print(f"Reason: {decisions['PLTR']['explain']}")
print(f"\nDIP Detected: {'[DIP]' in decisions['PLTR']['explain']}")
