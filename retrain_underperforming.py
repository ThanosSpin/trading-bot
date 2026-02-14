#!/usr/bin/env python
"""
Retrain Underperforming Models
Automatically identifies and retrains models below accuracy threshold.

Usage:
    python retrain_underperforming.py                    # Auto-detect from diagnostics
    python retrain_underperforming.py --threshold 0.52   # Custom accuracy threshold
    python retrain_underperforming.py --symbols NVDA SPY # Force specific symbols
    python retrain_underperforming.py --tune             # Include hyperparameter tuning
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

# Import your existing modules
try:
    from data_loader import fetch_historical_data
    from features import build_intraday_features, clean_columns
    from predictive_model.model_xgb import train_model
    from config.config import SYMBOL
    
    # Try to import directory configs, use defaults if missing
    try:
        from config import LOGS_DIR, MODEL_DIR
    except ImportError:
        LOGS_DIR = 'logs'
        MODEL_DIR = 'models'
        print(f"[WARN] Using default directories: LOGS_DIR={LOGS_DIR}, MODEL_DIR={MODEL_DIR}")
        
except ImportError as e:
    print(f"[ERROR] Missing required module: {e}")
    print("Ensure data_loader.py, features.py, model_xgb.py, and config.py exist")
    sys.exit(1)

# Ensure directories exist
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# PERFORMANCE ANALYSIS
# ============================================================

def analyze_model_performance(symbol: str, model_type: str) -> dict:
    """
    Analyze actual performance from prediction logs.
    
    Returns:
        dict with 'accuracy', 'total_predictions', 'has_outcomes'
    """
    log_file = os.path.join(LOGS_DIR, f"predictions_{symbol}_{model_type}.csv")
    
    if not os.path.exists(log_file):
        return {'accuracy': None, 'total_predictions': 0, 'has_outcomes': False}
    
    try:
        df = pd.read_csv(log_file)
        
        if len(df) == 0:
            return {'accuracy': None, 'total_predictions': 0, 'has_outcomes': False}
        
        # Check if outcomes exist
        if 'actual_outcome' not in df.columns:
            return {'accuracy': None, 'total_predictions': len(df), 'has_outcomes': False}
        
        # Filter to rows with outcomes
        df_eval = df[df['actual_outcome'].notna()].copy()
        
        if len(df_eval) == 0:
            return {'accuracy': None, 'total_predictions': len(df), 'has_outcomes': False}
        
        # Calculate accuracy
        df_eval['predicted_outcome'] = (df_eval['predicted_prob'] > 0.5).astype(int)
        accuracy = (df_eval['predicted_outcome'] == df_eval['actual_outcome']).mean()
        
        return {
            'accuracy': accuracy,
            'total_predictions': len(df),
            'evaluated_predictions': len(df_eval),
            'has_outcomes': True
        }
        
    except Exception as e:
        print(f"[ERROR] analyze_model_performance({symbol}, {model_type}): {e}")
        return {'accuracy': None, 'total_predictions': 0, 'has_outcomes': False}


def identify_underperformers(symbols: list = None, threshold: float = 0.52) -> list:
    """
    Identify models that need retraining.
    
    Args:
        symbols: List of symbols to check (None = all from config)
        threshold: Minimum acceptable accuracy
    
    Returns:
        List of (symbol, model_type, accuracy) tuples for underperformers
    """
    if symbols is None:
        symbols = SYMBOL
    
    model_types = ['intraday_mom', 'intraday_mr']
    underperformers = []
    
    print(f"\n{'='*70}")
    print(f"🔍 ANALYZING MODEL PERFORMANCE")
    print(f"{'='*70}")
    print(f"Threshold: {threshold*100:.1f}% accuracy")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"{'='*70}\n")
    
    for symbol in symbols:
        for model_type in model_types:
            perf = analyze_model_performance(symbol, model_type)
            
            if not perf['has_outcomes']:
                print(f"⏭️  {symbol:6} {model_type:15} - No outcomes yet")
                continue
            
            acc = perf['accuracy']
            
            if acc is None:
                print(f"⏭️  {symbol:6} {model_type:15} - No data")
                continue
            
            # Check if underperforming
            if acc < threshold:
                print(f"⚠️  {symbol:6} {model_type:15} - {acc*100:5.1f}% (BELOW THRESHOLD)")
                underperformers.append((symbol, model_type, acc))
            else:
                print(f"✅ {symbol:6} {model_type:15} - {acc*100:5.1f}% (OK)")
    
    print(f"\n{'='*70}")
    print(f"Found {len(underperformers)} model(s) needing retraining")
    print(f"{'='*70}\n")
    
    return underperformers


# ============================================================
# RETRAINING LOGIC
# ============================================================

def retrain_model(symbol: str, model_type: str, tune: bool = False, lookback_days: int = 60):
    """
    Retrain a specific model with fresh data.
    
    Args:
        symbol: Stock symbol
        model_type: 'intraday_mom' or 'intraday_mr'
        tune: If True, perform hyperparameter tuning (passed to train_model)
        lookback_days: Days of historical data to use (max 60 for intraday)
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*70}")
    print(f"🔄 RETRAINING: {symbol} - {model_type}")
    print(f"{'='*70}")
    
    # ✅ Validate lookback for intraday data
    if 'intraday' in model_type and lookback_days > 60:
        print(f"[WARN] Yahoo Finance limits intraday data to 60 days")
        print(f"[WARN] Reducing lookback from {lookback_days} to 60 days")
        lookback_days = 60
    
    try:
        # Fetch fresh data
        print(f"[DATA] Fetching {lookback_days} days of intraday data...")
        df = fetch_historical_data(
            symbol,
            period=f"{lookback_days}d",
            interval="15m"
        )
        
        if df is None or len(df) == 0:
            print(f"[ERROR] No data available for {symbol}")
            return False
        
        print(f"[DATA] Loaded {len(df)} bars")
        
        # Clean columns
        df = clean_columns(df)
        
        # Train model using your existing training function
        print(f"[TRAIN] Training {model_type} model...")
        
        if tune:
            print("[TRAIN] ⚠️  Note: Hyperparameter tuning must be implemented in train_model()")
            print("[TRAIN] Current script will use default parameters from model_xgb.py")
        
        # Your train_model function handles everything
        artifact = train_model(df, symbol=symbol, mode=model_type)
        
        if artifact is None:
            print(f"[ERROR] Training failed for {symbol} {model_type}")
            return False
        
        # Save the model
        model_path = os.path.join(MODEL_DIR, f"{symbol}_{model_type}_xgb.pkl")
        joblib.dump(artifact, model_path)
        
        print(f"[SUCCESS] Model saved: {model_path}")
        
        # Display metrics
        metrics = artifact.get('metrics', {})
        print(f"\n[METRICS] New model performance:")
        print(f"  Accuracy:  {metrics.get('accuracy', 0):.3f}")
        if metrics.get('logloss') is not None:
            print(f"  Log Loss:  {metrics.get('logloss'):.3f}")
        if metrics.get('rocauc') is not None:
            print(f"  ROC AUC:   {metrics.get('rocauc'):.3f}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Retraining failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# MAIN WORKFLOW
# ============================================================

def retrain_underperformers(
    symbols: list = None,
    threshold: float = 0.52,
    tune: bool = False,
    lookback_days: int = 60,  # ✅ Changed default to 60
    force_symbols: list = None
):
    """
    Main workflow: identify and retrain underperforming models.
    
    Args:
        symbols: Symbols to analyze (None = all from config)
        threshold: Accuracy threshold for retraining
        tune: Enable hyperparameter tuning
        lookback_days: Days of data for retraining (max 60 for intraday)
        force_symbols: Force retrain these (symbol, model_type) tuples
    """
    print(f"\n{'='*70}")
    print(f"🤖 AUTOMATED MODEL RETRAINING")
    print(f"{'='*70}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Threshold: {threshold*100:.1f}%")
    print(f"Tuning: {'Enabled' if tune else 'Disabled'}")
    print(f"Lookback: {lookback_days} days")
    print(f"{'='*70}\n")
    
    # Identify underperformers (unless forcing specific models)
    if force_symbols:
        to_retrain = [(s, m, None) for s, m in force_symbols]
        print(f"[FORCED] Retraining {len(to_retrain)} model(s):")
        for s, m, _ in to_retrain:
            print(f"  - {s} {m}")
    else:
        to_retrain = identify_underperformers(symbols, threshold)
    
    if len(to_retrain) == 0:
        print("\n✅ All models performing above threshold!")
        print("💡 Consider:")
        print("   - Lowering threshold to retrain marginal models")
        print("   - Using --force to retrain specific models")
        return
    
    # Ask for confirmation
    print(f"\n{'='*70}")
    print(f"⚠️  About to retrain {len(to_retrain)} model(s)")
    print(f"{'='*70}")
    
    if not force_symbols:  # Skip confirmation if forced
        response = input("\nProceed? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Retrain each model
    results = []
    
    for symbol, model_type, old_acc in to_retrain:
        success = retrain_model(symbol, model_type, tune=tune, lookback_days=lookback_days)
        results.append((symbol, model_type, old_acc, success))
        print()  # Spacing
    
    # Summary report
    print(f"\n{'='*70}")
    print(f"📊 RETRAINING SUMMARY")
    print(f"{'='*70}\n")
    
    successful = sum(1 for _, _, _, s in results if s)
    failed = len(results) - successful
    
    for symbol, model_type, old_acc, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        acc_str = f"(was {old_acc*100:.1f}%)" if old_acc else ""
        print(f"{status:12} {symbol:6} {model_type:15} {acc_str}")
    
    print(f"\n{'='*70}")
    print(f"✅ Successful: {successful}/{len(results)}")
    print(f"❌ Failed:     {failed}/{len(results)}")
    print(f"{'='*70}\n")
    
    if successful > 0:
        print("💡 Next steps:")
        print("   1. Wait for new predictions to accumulate")
        print("   2. Run outcome_tracker.py to fill in results")
        print("   3. Run diagnose_intraday_models.py to verify improvement")
        print("\n   python outcome_tracker.py")
        print("   python diagnose_intraday_models.py")


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Automatically retrain underperforming models"
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=None,
        help='Symbols to analyze (default: all from config)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.52,
        help='Accuracy threshold for retraining (default: 0.52)'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable hyperparameter tuning (slower but better)'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=60,  # ✅ Changed default to 60
        help='Days of historical data to use (default: 60, max for intraday)'
    )
    parser.add_argument(
        '--force',
        nargs='+',
        metavar='SYMBOL:MODEL',
        help='Force retrain specific models (e.g., NVDA:intraday_mom SPY:intraday_mom)'
    )
    
    args = parser.parse_args()
    
    # Parse forced models if provided
    force_symbols = None
    if args.force:
        force_symbols = []
        for item in args.force:
            try:
                symbol, model = item.split(':')
                force_symbols.append((symbol.upper(), model))
            except ValueError:
                print(f"[ERROR] Invalid format: {item}")
                print("Use format: SYMBOL:MODEL (e.g., NVDA:intraday_mom)")
                sys.exit(1)
    
    try:
        retrain_underperformers(
            symbols=args.symbols,
            threshold=args.threshold,
            tune=args.tune,
            lookback_days=args.lookback,
            force_symbols=force_symbols
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
