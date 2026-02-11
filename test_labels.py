# Check if labels are inverted
import pandas as pd

df = pd.read_csv('logs/predictions_NVDA_intraday_mom.csv')
df = df.dropna(subset=['predicted_prob', 'actual_outcome', 'price'])

# Calculate next-bar return
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')
df['next_price'] = df['price'].shift(-1)
df['actual_return'] = (df['next_price'] - df['price']) / df['price']

# Check correlation
correlation = df['predicted_prob'].corr(df['actual_return'])
print(f"Correlation (predicted_prob vs actual_return): {correlation:.3f}")

if correlation < -0.1:
    print("❌ INVERTED: Model predicts OPPOSITE of reality")
elif correlation > 0.1:
    print("✅ CORRECT: Model predicts in right direction")
else:
    print("⚠️ NO SIGNAL: Model has no predictive power")
