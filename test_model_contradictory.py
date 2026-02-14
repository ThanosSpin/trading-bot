from model_xgb import compute_signals

sig = compute_signals('NVDA')

print(f"Daily: {sig.get('daily_prob'):.2f}")
print(f"Intraday: {sig.get('intraday_prob'):.2f}")
print(f"Weight original: {sig.get('intraday_weight_original'):.2f}")
print(f"Weight used: {sig.get('intraday_weight_used'):.2f}")
print(f"Contradiction: {sig.get('contradiction_detected')}")
print(f"Final: {sig.get('final_prob'):.2f}")

# Verify logic
if sig.get('contradiction_detected'):
    assert sig.get('intraday_weight_used') == 0.0, "Weight not zeroed!"
    print("✅ Contradiction handling CORRECT")