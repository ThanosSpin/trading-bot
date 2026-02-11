#!/usr/bin/env python
"""
Retrain intraday_mom models with proper regime filtering.
Only trains on momentum regime periods to prevent mean-reversion learning.

NOTE: Yahoo Finance limits 15min data to 60 days max
"""

import pandas as pd
import sys
import os

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import fetch_historical_data
from model_xgb import train_model


def retrain_momentum_model(symbol, period="60d"):  # ← Changed from 6mo to 60d
    """Retrain with momentum-period filtering"""
    
    print(f"\n{'='*60}")
    print(f"Retraining {symbol} intraday_mom with regime filtering")
    print(f"{'='*60}")
    
    # Fetch data (max 60 days for 15min data)
    df = fetch_historical_data(symbol, period=period, interval="15m")
    
    if df is None or len(df) < 500:
        print(f"❌ Insufficient data for {symbol} ({len(df) if df is not None else 0} bars)")
        print(f"   Need at least 500 bars for training")
        return
    
    print(f"📊 Loaded {len(df)} bars")
    
    # Calculate momentum and volatility metrics
    df['mom_12'] = df['Close'].pct_change(12).abs()  # 3-hour momentum magnitude
    df['vol_12'] = df['Close'].pct_change().rolling(12).std()
    
    # Define momentum regime thresholds
    # For individual stocks: use 70th percentile
    # For SPY: use 60th percentile (less volatile)
    if symbol == "SPY":
        mom_percentile = 0.60
        vol_percentile = 0.60
    else:
        mom_percentile = 0.70
        vol_percentile = 0.70
    
    mom_threshold = df['mom_12'].quantile(mom_percentile)
    vol_threshold = df['vol_12'].quantile(vol_percentile)
    
    print(f"📈 Momentum threshold ({mom_percentile:.0%} percentile): {mom_threshold:.4f}")
    print(f"📈 Volatility threshold ({vol_percentile:.0%} percentile): {vol_threshold:.4f}")
    
    # Filter to momentum periods ONLY
    # Keep bars where EITHER momentum is high OR volatility is high
    df_mom = df[
        (df['mom_12'] > mom_threshold) | 
        (df['vol_12'] > vol_threshold)
    ].copy()
    
    kept_pct = 100 * len(df_mom) / len(df)
    print(f"✅ Kept {len(df_mom)} momentum bars ({kept_pct:.1f}% of data)")
    
    # Check if we have enough samples
    min_samples = 200
    if len(df_mom) < min_samples:
        print(f"⚠️ Too few momentum samples ({len(df_mom)}) - need {min_samples}+")
        print(f"   Trying with lower percentile threshold...")
        
        # Retry with lower threshold
        mom_threshold = df['mom_12'].quantile(0.50)
        vol_threshold = df['vol_12'].quantile(0.50)
        
        df_mom = df[
            (df['mom_12'] > mom_threshold) | 
            (df['vol_12'] > vol_threshold)
        ].copy()
        
        kept_pct = 100 * len(df_mom) / len(df)
        print(f"📊 Retry: Kept {len(df_mom)} momentum bars ({kept_pct:.1f}% of data)")
        
        if len(df_mom) < min_samples:
            print(f"❌ Still insufficient data - cannot retrain {symbol}")
            return
    
    # Train model on filtered data
    print(f"\n🔄 Training momentum model on filtered data...")
    
    try:
        # Call your existing training function
        train_model(
            df_mom, 
            symbol=symbol, 
            mode="intraday_mom",
            use_multiclass=False
        )
        
        print(f"✅ {symbol} intraday_mom retrained with regime filtering")
        print(f"   Model saved to: models/{symbol}_intraday_mom_xgb.pkl\n")
        
    except Exception as e:
        print(f"❌ Training failed for {symbol}: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Retrain all symbols"""
    symbols = ['NVDA', 'AAPL', 'ABBV', 'PLTR', 'SPY']
    
    print(f"\n{'='*60}")
    print(f"🔄 MOMENTUM MODEL RETRAINING")
    print(f"{'='*60}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: 60 days (Yahoo Finance limit for 15min data)")
    print(f"Strategy: Train only on high-momentum periods")
    print(f"{'='*60}\n")
    
    success_count = 0
    failed_symbols = []
    
    for sym in symbols:
        try:
            retrain_momentum_model(sym, period="60d")
            success_count += 1
        except Exception as e:
            print(f"❌ Failed to retrain {sym}: {e}\n")
            failed_symbols.append(sym)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 RETRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully retrained: {success_count}/{len(symbols)} symbols")
    
    if failed_symbols:
        print(f"❌ Failed: {', '.join(failed_symbols)}")
    else:
        print(f"🎉 All models retrained successfully!")
    
    print(f"\n⚠️  NEXT STEPS:")
    print(f"1. Remove the inversion hotfix from model_xgb.py")
    print(f"2. Test models with: python main.py")
    print(f"3. Validate with: python diagnose_intraday_models.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
