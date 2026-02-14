# retrain_intraday_dual.py
"""
Weekend intraday dual-model retraining script.

Trains both regime models (mean-reversion + momentum) for all symbols.
Enhanced with Week 1 optimizations:
- Multi-class targets
- Enhanced indicators
- Proper error handling
"""

import os
import sys
from datetime import datetime
import joblib
import traceback

from predictive_model.data_loader import fetch_historical_data
from predictive_model.features import _clean_columns
from predictive_model.model_xgb import train_model, MODEL_DIR
from config.config import TRAIN_SYMBOLS, USE_MULTICLASS_MODELS


# ============================================================
# CONFIGURATION
# ============================================================

INTRADAY_LOOKBACK_DAYS = 60
INTRADAY_INTERVAL = "15m"

# ✅ Week 1 enhancements
USE_MULTICLASS = USE_MULTICLASS_MODELS  # 5-class targets instead of binary
FORCE_RETRAIN = True   # Skip model existence checks


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _save_artifact(artifact, symbol: str, mode: str):
    """Save model artifact with verification."""
    model_dir = MODEL_DIR if MODEL_DIR else "models"
    os.makedirs(model_dir, exist_ok=True)
    
    path = os.path.join(model_dir, f"{symbol}_{mode}_xgb.pkl")
    abs_path = os.path.abspath(path)
    
    try:
        joblib.dump(artifact, path)
        
        # Verify save
        if os.path.exists(path):
            file_size = os.path.getsize(path) / 1024  # KB
            print(f"✅ Saved {symbol} {mode}: {abs_path} ({file_size:.1f} KB)")
            return True
        else:
            print(f"❌ Failed to save {symbol} {mode} - file not found after save")
            return False
            
    except Exception as e:
        print(f"❌ Error saving {symbol} {mode}: {e}")
        traceback.print_exc()
        return False


def _train_regime_model(df, symbol: str, mode: str, use_multiclass: bool = True):
    """
    Train a single regime model with error handling.
    
    Args:
        df: Raw OHLCV data
        symbol: Stock symbol
        mode: 'intraday_mr' or 'intraday_mom'
        use_multiclass: If True, use 5-class targets
    
    Returns:
        artifact dict or None if failed
    """
    print(f"\n{'='*60}")
    print(f"Training {symbol} - {mode.upper()}")
    print(f"{'='*60}")
    
    try:
        artifact = train_model(
            df, 
            symbol=symbol, 
            mode=mode,
            use_multiclass=use_multiclass
        )
        
        if artifact is None:
            print(f"❌ Training returned None for {symbol} {mode}")
            return None
        
        # Validate artifact
        required_keys = ['model', 'features', 'metrics', 'trained_at']
        missing_keys = [k for k in required_keys if k not in artifact]
        
        if missing_keys:
            print(f"⚠️ Warning: Artifact missing keys: {missing_keys}")
        
        # Print key metrics
        metrics = artifact.get('metrics', {})
        if metrics:
            print(f"\n📊 Model Metrics:")
            for key in ['accuracy', 'roc_auc', 'logloss', 'f1', 'brier_score']:
                if key in metrics and metrics[key] is not None:
                    print(f"  {key:15} {metrics[key]:.4f}")
        
        return artifact
        
    except ValueError as e:
        # Expected errors (insufficient data, class diversity, etc.)
        error_msg = str(e)
        if 'Insufficient class diversity' in error_msg or 'not enough filtered data' in error_msg:
            print(f"⚠️ Expected error for {symbol} {mode}:")
            print(f"   {error_msg}")
            print(f"   This is normal for symbols with insufficient data in this regime")
        else:
            print(f"⚠️ ValueError for {symbol} {mode}: {e}")
        return None
        
    except Exception as e:
        # Unexpected errors
        print(f"❌ Unexpected error training {symbol} {mode}:")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def _print_summary(results: dict):
    """Print training summary."""
    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*60}")
    
    total_models = len(results)
    successful = sum(1 for v in results.values() if v['success'])
    failed = total_models - successful
    
    print(f"Total models:     {total_models}")
    print(f"Successful:       {successful} ({successful/total_models*100:.1f}%)")
    print(f"Failed:           {failed} ({failed/total_models*100:.1f}%)")
    
    print(f"\nDetailed Results:")
    print(f"{'Symbol':<8} {'Mode':<15} {'Status':<10} {'Reason'}")
    print(f"{'-'*60}")
    
    for (sym, mode), info in sorted(results.items()):
        status = '✅ Success' if info['success'] else '❌ Failed'
        reason = info.get('reason', '')
        print(f"{sym:<8} {mode:<15} {status:<10} {reason}")
    
    print(f"{'='*60}\n")


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def main():
    """Main training function."""
    
    print(f"\n{'='*70}")
    print(f"🔄 WEEKEND INTRADAY DUAL-MODEL TRAINING")
    print(f"{'='*70}")
    print(f"📂 Model directory: {os.path.abspath(MODEL_DIR if MODEL_DIR else 'models')}")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Multi-class mode: {'Enabled (5 classes)' if USE_MULTICLASS else 'Disabled (binary)'}")
    
    # Get symbols
    symbols = TRAIN_SYMBOLS if isinstance(TRAIN_SYMBOLS, list) else [TRAIN_SYMBOLS]
    print(f"📌 Symbols ({len(symbols)}): {', '.join(symbols)}")
    print(f"{'='*70}\n")
    
    # Track results
    results = {}
    
    # Train each symbol
    for i, sym in enumerate(symbols, 1):
        print(f"\n{'#'*70}")
        print(f"# SYMBOL {i}/{len(symbols)}: {sym}")
        print(f"{'#'*70}")
        
        # Fetch data
        print(f"\n[DATA] Fetching {INTRADAY_LOOKBACK_DAYS}d of {INTRADAY_INTERVAL} data for {sym}...")
        
        try:
            df = fetch_historical_data(
                sym,
                period=f"{INTRADAY_LOOKBACK_DAYS}d",
                interval=INTRADAY_INTERVAL,
            )
        except Exception as e:
            print(f"❌ Data fetch failed for {sym}: {e}")
            results[(sym, 'intraday_mr')] = {'success': False, 'reason': 'Data fetch failed'}
            results[(sym, 'intraday_mom')] = {'success': False, 'reason': 'Data fetch failed'}
            continue
        
        if df is None or df.empty:
            print(f"❌ No intraday data for {sym}, skipping.")
            results[(sym, 'intraday_mr')] = {'success': False, 'reason': 'No data'}
            results[(sym, 'intraday_mom')] = {'success': False, 'reason': 'No data'}
            continue
        
        print(f"[DATA] ✅ Fetched {len(df)} bars for {sym}")
        
        # Clean columns
        df = _clean_columns(df)
        
        # ========================================
        # TRAIN MEAN-REVERSION MODEL
        # ========================================
        
        artifact_mr = _train_regime_model(
            df, 
            symbol=sym, 
            mode="intraday_mr",
            use_multiclass=USE_MULTICLASS
        )
        
        if artifact_mr:
            success = _save_artifact(artifact_mr, sym, "intraday_mr")
            results[(sym, 'intraday_mr')] = {
                'success': success,
                'reason': 'Saved successfully' if success else 'Save failed'
            }
        else:
            results[(sym, 'intraday_mr')] = {
                'success': False,
                'reason': 'Training failed'
            }
        
        # ========================================
        # TRAIN MOMENTUM MODEL
        # ========================================
        
        artifact_mom = _train_regime_model(
            df, 
            symbol=sym, 
            mode="intraday_mom",
            use_multiclass=USE_MULTICLASS
        )
        
        if artifact_mom:
            success = _save_artifact(artifact_mom, sym, "intraday_mom")
            results[(sym, 'intraday_mom')] = {
                'success': success,
                'reason': 'Saved successfully' if success else 'Save failed'
            }
        else:
            results[(sym, 'intraday_mom')] = {
                'success': False,
                'reason': 'Training failed'
            }
    
    # ========================================
    # PRINT SUMMARY
    # ========================================
    
    _print_summary(results)
    
    print(f"{'='*70}")
    print(f"🎉 Weekend intraday training complete!")
    print(f"📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error in main():")
        print(f"   {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
