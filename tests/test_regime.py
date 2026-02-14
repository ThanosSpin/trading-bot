from model_xgb import compute_signals

print("="*70)
print("TESTING REGIME DETECTION")
print("="*70)

for symbol in ["NVDA", "AAPL", "ABBV", "PLTR", "SPY"]:
    print(f"\n{'─'*70}")
    print(f"Testing {symbol}")
    print('─'*70)
    
    results = compute_signals(symbol, lookback_minutes=900)
    
    regime = results.get('intraday_regime', 'N/A')
    model = results.get('intraday_model_used', 'N/A')
    mom = results.get('intraday_mom')
    vol = results.get('intraday_vol')
    prob = results.get('intraday_prob')
    
    mom_str = f"{mom:.4f}" if mom else "N/A"
    vol_str = f"{vol:.5f}" if vol else "N/A"
    prob_str = f"{prob:.3f}" if prob else "N/A"
    
    print(f"Regime:    {regime}")
    print(f"Model:     {model}")
    print(f"Momentum:  {mom_str}")
    print(f"Volatility: {vol_str}")
    print(f"Prob:      {prob_str}")

print(f"\n{'='*70}")
print("✅ TEST COMPLETE")
print('='*70)
