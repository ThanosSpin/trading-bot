import pandas as pd
import numpy as np

# Load PLTR prediction logs
try:
    df = pd.read_csv('logs/predictions_PLTR_intraday_mom.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)  # Force UTC
    df = df.sort_values('timestamp')
    
    # Check recent predictions (last 24 hours)
    now_utc = pd.Timestamp.now(tz='UTC')
    recent = df[df['timestamp'] > now_utc - pd.Timedelta(days=1)]
    
    print(f"\n{'='*60}")
    print(f"PLTR intraday_mom Model Diagnostic")
    print(f"{'='*60}")
    print(f"Total predictions logged: {len(df)}")
    print(f"Recent (last 24h): {len(recent)}")
    
    if len(recent) > 0:
        print(f"\nRecent Prediction Statistics:")
        print(f"  Mean predicted prob: {recent['predicted_prob'].mean():.3f}")
        print(f"  Std dev: {recent['predicted_prob'].std():.3f}")
        print(f"  Min: {recent['predicted_prob'].min():.3f}")
        print(f"  Max: {recent['predicted_prob'].max():.3f}")
        print(f"  Median: {recent['predicted_prob'].median():.3f}")
        
        # Check for bias
        if recent['predicted_prob'].mean() > 0.65:
            print(f"\n⚠️ WARNING: Model is strongly bullish-biased (mean > 0.65)")
        elif recent['predicted_prob'].mean() < 0.35:
            print(f"\n⚠️ WARNING: Model is strongly bearish-biased (mean < 0.35)")
        
        # Check variance
        if recent['predicted_prob'].std() < 0.05:
            print(f"⚠️ WARNING: Low variance ({recent['predicted_prob'].std():.3f}) - model may be stuck")
        
        # Show actual outcomes if available
        if 'actual_outcome' in df.columns:
            valid_outcomes = recent.dropna(subset=['actual_outcome'])
            if len(valid_outcomes) > 0:
                print(f"\nActual Outcomes (where available):")
                print(f"  Count: {len(valid_outcomes)}")
                print(f"  Mean actual: {valid_outcomes['actual_outcome'].mean():.3f}")
                print(f"  Mean predicted: {valid_outcomes['predicted_prob'].mean():.3f}")
                
                # Calculate calibration error
                calib_error = abs(valid_outcomes['predicted_prob'].mean() - 
                                 valid_outcomes['actual_outcome'].mean())
                print(f"  Calibration error: {calib_error:.3f}")
                
                if calib_error > 0.15:
                    print(f"  ❌ POOR CALIBRATION (error > 0.15)")
        
        # Show last 10 predictions
        print(f"\nLast 10 Predictions:")
        print(recent[['timestamp', 'predicted_prob', 'price']].tail(10).to_string(index=False))
    
    else:
        print(f"\n⚠️ No predictions in last 24 hours")
    
    print(f"{'='*60}\n")

except FileNotFoundError:
    print("❌ File not found: logs/predictions_PLTR_intraday_mom.csv")
    print("   Run the bot first to generate prediction logs")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
