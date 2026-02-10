# Check if logged predictions actually worked
import pandas as pd

# Load PLTR prediction logs
df = pd.read_csv('logs/predictions_PLTR_intraday_mom.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# Check recent predictions (last 24 hours)
recent = df[df['timestamp'] > pd.Timestamp.now() - pd.Timedelta(days=1)]

print(f"Recent PLTR intraday_mom predictions:")
print(f"  Count: {len(recent)}")
print(f"  Mean predicted prob: {recent['predicted_prob'].mean():.3f}")
print(f"  Actual outcomes (if available): {recent['actual_outcome'].mean():.3f}")

# Check if model is just predicting high constantly
if recent['predicted_prob'].mean() > 0.65:
    print("⚠️ Model is biased bullish - predicting >65% on average")
