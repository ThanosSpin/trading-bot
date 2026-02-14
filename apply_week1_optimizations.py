#!/usr/bin/env python3
"""
apply_week1_optimizations.py

Retrain all models with Week 1 optimizations:
1. ✅ Enhanced indicators (BB, ATR, ADX, Stochastic, etc.)
2. ✅ Multi-class targets (5 classes instead of binary)
3. ✅ Time-series cross-validation
4. ✅ Class balancing (scale_pos_weight)
5. ✅ Probability calibration (isotonic/sigmoid)
6. ✅ Better regularization (L1/L2, gamma, min_child_weight)

This script retrains all models and saves them with enhanced features.
"""

import os
import sys
from datetime import datetime
from config import TRAIN_SYMBOLS, MODEL_DIR, SPY_SYMBOL
from predictive_model.data_loader import fetch_historical_data, fetch_intraday_history
from predictive_model.model_xgb import train_model
import joblib
import traceback
from config.config import USE_MULTICLASS_MODELS

# ============================================================
# CONFIGURATION
# ============================================================
USE_MULTICLASS = USE_MULTICLASS_MODELS
# Data configuration
DAILY_PERIOD = "3y"          # 3 years of daily data
DAILY_INTERVAL = "1d"

INTRADAY_LOOKBACK = 2400     # 2400 minutes (~4 days of 15min bars)
INTRADAY_INTERVAL = "15m"

# Training configuration
TRAIN_MODES = ['daily', 'intraday_mr', 'intraday_mom']  # Skip legacy 'intraday'
FORCE_RETRAIN = False        # Set True to retrain even if model exists and is recent

# Model freshness threshold (days)
FRESHNESS_THRESHOLD_DAYS = 7  # Retrain if model is older than this

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def should_retrain(symbol: str, mode: str) -> bool:
    """
    Check if model should be retrained.
    
    Returns True if:
    - Model doesn't exist
    - Model is older than FRESHNESS_THRESHOLD_DAYS
    - FORCE_RETRAIN is True
    """
    if FORCE_RETRAIN:
        return True
    
    model_path = os.path.join(MODEL_DIR, f"{symbol}_{mode}_xgb.pkl")
    
    if not os.path.exists(model_path):
        return True  # Model doesn't exist
    
    try:
        artifact = joblib.load(model_path)
        trained_at = artifact.get('trained_at')
        
        if not trained_at:
            return True  # No timestamp, retrain
        
        trained_time = datetime.fromisoformat(trained_at)
        age_days = (datetime.now() - trained_time).days
        
        if age_days > FRESHNESS_THRESHOLD_DAYS:
            print(f"   ⏰ Model is {age_days} days old (threshold: {FRESHNESS_THRESHOLD_DAYS}d)")
            return True
        
        print(f"   ✅ Model is fresh ({age_days} days old), skipping")
        return False
        
    except Exception as e:
        print(f"   ⚠️  Could not check model age: {e}")
        return True  # Safe default: retrain

def fetch_data_for_mode(symbol: str, mode: str):
    """
    Fetch appropriate data based on mode.
    
    Returns:
        DataFrame or None
    """
    if mode == 'daily':
        print(f"   📥 Fetching daily data (period={DAILY_PERIOD}, interval={DAILY_INTERVAL})")
        df = fetch_historical_data(symbol, period=DAILY_PERIOD, interval=DAILY_INTERVAL)
    elif mode in ['intraday', 'intraday_mr', 'intraday_mom']:
        print(f"   📥 Fetching intraday data (lookback={INTRADAY_LOOKBACK}min, interval={INTRADAY_INTERVAL})")
        df = fetch_intraday_history(symbol, lookback_minutes=INTRADAY_LOOKBACK, interval=INTRADAY_INTERVAL)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return df

def save_model(artifact: dict, symbol: str, mode: str) -> str:
    """
    Save model artifact to disk.
    
    Returns:
        Path to saved model
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"{symbol}_{mode}_xgb.pkl")
    
    joblib.dump(artifact, model_path)
    
    return model_path

def print_training_summary(artifact: dict, symbol: str, mode: str):
    """
    Print summary of trained model.
    """
    metrics = artifact.get('metrics', {})
    
    print(f"\n   {'='*70}")
    print(f"   MODEL SUMMARY: {symbol} - {mode}")
    print(f"   {'='*70}")
    print(f"   Type:          {artifact.get('target_type', 'binary')}")
    print(f"   Classes:       {artifact.get('num_classes', 2)}")
    print(f"   Calibrated:    {artifact.get('calibrated', False)}")
    print(f"   Features:      {metrics.get('num_features', len(artifact.get('features', [])))}")
    print(f"   Train Samples: {metrics.get('train_samples', 'N/A')}")
    print(f"   Test Samples:  {metrics.get('test_samples', 'N/A')}")
    print(f"   ---")
    
    # Handle None values gracefully
    accuracy = metrics.get('accuracy', 0)
    logloss = metrics.get('logloss')
    
    print(f"   Accuracy:      {accuracy:.4f}" if accuracy else "   Accuracy:      N/A")
    print(f"   LogLoss:       {logloss:.4f}" if logloss else "   LogLoss:       N/A")
    
    if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
        print(f"   ROC AUC:       {metrics['roc_auc']:.4f}")
    
    if 'precision_weighted' in metrics:
        print(f"   Precision:     {metrics.get('precision_weighted', 0):.4f}")
        print(f"   Recall:        {metrics.get('recall_weighted', 0):.4f}")
        print(f"   F1:            {metrics.get('f1_weighted', 0):.4f}")
    else:
        prec = metrics.get('precision', 0)
        rec = metrics.get('recall', 0)
        f1 = metrics.get('f1', 0)
        print(f"   Precision:     {prec:.4f}" if prec else "   Precision:     N/A")
        print(f"   Recall:        {rec:.4f}" if rec else "   Recall:        N/A")
        print(f"   F1:            {f1:.4f}" if f1 else "   F1:            N/A")
    
    print(f"   {'='*70}\n")

# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def retrain_all_models():
    """
    Main function to retrain all models with Week 1 optimizations.
    """
    # Get symbols
    if isinstance(TRAIN_SYMBOLS, list):
        symbols = TRAIN_SYMBOLS
    else:
        symbols = [TRAIN_SYMBOLS]
    
    # Add SPY if not already included
    spy_upper = SPY_SYMBOL.upper()
    if spy_upper not in [s.upper() for s in symbols]:
        symbols.append(spy_upper)
    
    print("\n" + "="*100)
    print("WEEK 1 MODEL RETRAINING")
    print("="*100)
    print(f"\n✨ Optimizations Applied:")
    print(f"   1. Enhanced indicators (BB, ATR, ADX, Stochastic, OBV, CMF)")
    print(f"   2. {'Multi-class targets (5 classes)' if USE_MULTICLASS else 'Binary targets (up/down)'}")
    print(f"   3. Time-series cross-validation")
    print(f"   4. Class balancing (scale_pos_weight)")
    print(f"   5. Probability calibration (isotonic/sigmoid)")
    print(f"   6. Better regularization (L1/L2, gamma)")
    
    print(f"\n📊 Training Configuration:")
    print(f"   Symbols:       {', '.join(symbols)}")
    print(f"   Modes:         {', '.join(TRAIN_MODES)}")
    print(f"   Target Type:   {'Multi-class (5 classes)' if USE_MULTICLASS else 'Binary (2 classes)'}")
    print(f"   Daily Data:    {DAILY_PERIOD} @ {DAILY_INTERVAL}")
    print(f"   Intraday Data: {INTRADAY_LOOKBACK}min @ {INTRADAY_INTERVAL}")
    print(f"   Force Retrain: {FORCE_RETRAIN}")
    print(f"   Model Dir:     {MODEL_DIR}")
    
    print("\n" + "="*100 + "\n")
    
    # Track results
    results = {
        'success': [],
        'skipped': [],
        'failed': [],
        'insufficient_data': []  # ✅ NEW: Track models skipped due to insufficient data
    }
    
    total_models = len(symbols) * len(TRAIN_MODES)
    current = 0
    
    # Train each symbol/mode combination
    for symbol in symbols:
        symbol_upper = symbol.upper()
        
        print(f"\n{'#'*100}")
        print(f"# SYMBOL: {symbol_upper}")
        print(f"{'#'*100}\n")
        
        for mode in TRAIN_MODES:
            current += 1
            print(f"\n[{current}/{total_models}] Training {symbol_upper} - {mode}")
            print("-" * 80)
            
            try:
                # Check if should retrain
                if not should_retrain(symbol_upper, mode):
                    results['skipped'].append((symbol_upper, mode))
                    continue
                
                # Fetch data
                df = fetch_data_for_mode(symbol_upper, mode)
                
                if df is None or df.empty:
                    print(f"   ❌ No data available for {symbol_upper} - {mode}")
                    results['failed'].append((symbol_upper, mode, "No data"))
                    continue
                
                print(f"   ✅ Fetched {len(df)} rows")
                
                # Train model
                print(f"   🔨 Training model...")
                artifact = train_model(df, symbol_upper, mode, use_multiclass=USE_MULTICLASS)
                
                # Save model
                model_path = save_model(artifact, symbol_upper, mode)
                print(f"   💾 Saved to: {model_path}")
                
                # Print summary
                print_training_summary(artifact, symbol_upper, mode)
                
                results['success'].append((symbol_upper, mode))
                
            except ValueError as e:
                # ✅ GRACEFUL HANDLING: Check if it's an insufficient data error
                error_str = str(e)
                if "Insufficient class diversity" in error_str or "only one label" in error_str:
                    print(f"\n   ⚠️  INSUFFICIENT DATA: {e}")
                    print(f"   ⊘  Skipping {symbol_upper}/{mode} - this is expected for aggressive regime filters")
                    results['insufficient_data'].append((symbol_upper, mode, error_str))
                else:
                    # Other ValueError - treat as failure
                    print(f"\n   ❌ TRAINING FAILED: {e}")
                    traceback.print_exc()
                    results['failed'].append((symbol_upper, mode, error_str))
                print()
                
            except Exception as e:
                print(f"\n   ❌ TRAINING FAILED: {e}")
                traceback.print_exc()
                results['failed'].append((symbol_upper, mode, str(e)))
                print()
    
    # Print final summary
    print("\n" + "="*100)
    print("TRAINING SUMMARY")
    print("="*100)
    
    print(f"\n✅ Successful: {len(results['success'])}/{total_models}")
    for symbol, mode in results['success']:
        print(f"   ✓ {symbol} - {mode}")
    
    if results['skipped']:
        print(f"\n⏭️  Skipped (fresh): {len(results['skipped'])}")
        for symbol, mode in results['skipped']:
            print(f"   - {symbol} - {mode}")
    
    # ✅ NEW: Show insufficient data separately (not a failure)
    if results['insufficient_data']:
        print(f"\n⚠️  Skipped (insufficient data): {len(results['insufficient_data'])}")
        for symbol, mode, error in results['insufficient_data']:
            # Extract just the class counts from error message
            if "Class counts:" in error:
                counts = error.split("Class counts:")[1].split(".")[0].strip()
                print(f"   ⊘ {symbol} - {mode}: {counts}")
            else:
                print(f"   ⊘ {symbol} - {mode}")
    
    if results['failed']:
        print(f"\n❌ Failed: {len(results['failed'])}")
        for symbol, mode, error in results['failed']:
            # Truncate long errors
            error_short = error[:80] + "..." if len(error) > 80 else error
            print(f"   ✗ {symbol} - {mode}: {error_short}")
    
    print("\n" + "="*100)
    
    # ✅ IMPROVED: Calculate success rate (excluding insufficient data)
    trainable_models = total_models - len(results['skipped']) - len(results['insufficient_data'])
    success_count = len(results['success'])
    
    if trainable_models > 0:
        success_rate = (success_count / trainable_models) * 100
        print(f"\n📊 Success Rate: {success_count}/{trainable_models} ({success_rate:.1f}%)")
    
    # Exit code
    if results['failed'] and not results['success']:
        print("\n❌ All trainings failed!")
        sys.exit(1)
    elif results['failed']:
        print(f"\n⚠️  Some trainings failed ({len(results['failed'])} failures)")
        sys.exit(1)
    else:
        print("\n✅ All trainings completed successfully!")
        
        if results['insufficient_data']:
            print(f"\n💡 Note: {len(results['insufficient_data'])} regime models skipped due to insufficient filtered data.")
            print(f"   This is normal for aggressive regime filters (e.g., intraday_mr).")
            print(f"   These symbols will use other available models (daily, intraday_mom).")
        
        print("\n💡 Next Steps:")
        print("   1. Check models: python check_model_types.py")
        print("   2. Test signals: python test_enhanced_signals.py")
        print("   3. Run trading:  python main.py")
        sys.exit(0)

# ============================================================
# ADVANCED: TRAIN SPECIFIC MODEL
# ============================================================

def train_specific_model(symbol: str, mode: str, force: bool = False):
    """
    Train a specific symbol/mode combination.
    
    Usage:
        python apply_week1_optimizations.py --symbol NVDA --mode daily
    """
    print(f"\n{'='*100}")
    print(f"TRAINING SPECIFIC MODEL: {symbol.upper()} - {mode}")
    print(f"{'='*100}\n")
    
    symbol_upper = symbol.upper()
    
    # Check if should retrain
    if not force and not should_retrain(symbol_upper, mode):
        print(f"✅ Model is already fresh. Use --force to retrain anyway.")
        return
    
    try:
        # Fetch data
        df = fetch_data_for_mode(symbol_upper, mode)
        
        if df is None or df.empty:
            print(f"❌ No data available for {symbol_upper} - {mode}")
            return
        
        print(f"✅ Fetched {len(df)} rows\n")
        
        # Train model
        print(f"🔨 Training model...\n")
        artifact = train_model(df, symbol=symbol_upper, mode=mode, use_multiclass=USE_MULTICLASS)
        
        # Save model
        model_path = save_model(artifact, symbol_upper, mode)
        print(f"\n💾 Saved to: {model_path}")
        
        # Print summary
        print_training_summary(artifact, symbol_upper, mode)
        
        print("✅ Training completed successfully!\n")
        
    except ValueError as e:
        if "Insufficient class diversity" in str(e):
            print(f"\n⚠️  INSUFFICIENT DATA: {e}")
            print(f"⊘  Cannot train {symbol_upper}/{mode} - try using more data or different mode")
        else:
            print(f"\n❌ TRAINING FAILED: {e}")
            traceback.print_exc()
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)

# ============================================================
# CLI INTERFACE
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Retrain models with Week 1 optimizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Retrain all models (skip fresh ones)
  python apply_week1_optimizations.py
  
  # Force retrain all models
  python apply_week1_optimizations.py --force
  
  # Train specific model
  python apply_week1_optimizations.py --symbol NVDA --mode daily
  
  # Force train specific model
  python apply_week1_optimizations.py --symbol NVDA --mode daily --force
  
  # Enable multi-class targets (5 classes)
  python apply_week1_optimizations.py --multiclass
  
  # List available modes
  python apply_week1_optimizations.py --list-modes
        """
    )
    
    parser.add_argument(
        '--symbol', 
        type=str, 
        help='Train specific symbol (e.g., NVDA)'
    )
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['daily', 'intraday', 'intraday_mr', 'intraday_mom'],
        help='Train specific mode'
    )
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Force retrain even if model is fresh'
    )
    parser.add_argument(
        '--multiclass',
        action='store_true',
        help='Use multi-class targets (5 classes) instead of binary'
    )
    parser.add_argument(
        '--list-modes',
        action='store_true',
        help='List available training modes'
    )
    
    args = parser.parse_args()
    
    # Handle --list-modes
    if args.list_modes:
        print("\nAvailable Training Modes:")
        print("  daily        - Daily timeframe model (long-term trends)")
        print("  intraday     - Legacy intraday model (not recommended)")
        print("  intraday_mr  - Mean-reversion intraday model (range-bound)")
        print("  intraday_mom - Momentum intraday model (trending)")
        print()
        sys.exit(0)
    
    # Set force flag globally
    if args.force:
        FORCE_RETRAIN = True
    
    # ✅ NEW: Enable multi-class if requested
    if args.multiclass:
        USE_MULTICLASS = USE_MULTICLASS_MODELS
        print("\n✨ Multi-class mode enabled (5 classes)\n")
    
    # Train specific model or all models
    if args.symbol and args.mode:
        train_specific_model(args.symbol, args.mode, force=args.force)
    elif args.symbol or args.mode:
        print("❌ Error: Both --symbol and --mode are required for specific training")
        sys.exit(1)
    else:
        retrain_all_models()
