#!/usr/bin/env python3
# retrain_model.py
"""
LEGACY COMPATIBILITY WRAPPER

This script maintains backward compatibility with the old SHAP-based
feature selection workflow while using the new Week 1 optimizations.

⚠️ RECOMMENDATION: Use apply_week1_optimizations.py instead for:
   - Enhanced indicators (BB, ATR, ADX, Stochastic)
   - Multi-class targets (5 classes)
   - Better calibration and regularization
   - Automatic backup management

This script is kept for:
   - SHAP feature selection workflows
   - Projects that depend on the old interface
   - Gradual migration path
"""

import os
import sys
import glob
import shutil
from datetime import datetime

import joblib

# ✅ UPDATED: Use new imports
from data_loader import fetch_historical_data, fetch_intraday_history
from predictive_model.model_xgb import train_model, MODEL_DIR
from config.config import TRAIN_SYMBOLS, SHAP_TOP_N, USE_MULTICLASS_MODELS

# ✅ NEW: Check if SHAP is available
try:
    from feature_selection import select_features_with_shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️  SHAP not available - feature selection will be skipped")
    SHAP_AVAILABLE = False

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
MAX_BACKUPS = 6
USE_MULTICLASS = USE_MULTICLASS_MODELS  # ✅ NEW: Set to True to enable 5-class targets

# ✅ UPDATED: Match apply_week1_optimizations.py data settings
DAILY_PERIOD = "3y"  # Was: LOOKBACK_YEARS = 1
DAILY_INTERVAL = "1d"

INTRADAY_LOOKBACK = 2400  # Was: INTRADAY_LOOKBACK_DAYS = 60 (now in minutes)
INTRADAY_INTERVAL = "15m"


# ---------------------------------------------------------
# Backup functions (shared with apply_week1_optimizations.py)
# ---------------------------------------------------------

def save_model_with_backup(artifact, symbol, mode: str = "daily"):
    """
    Save active model artifact and maintain monthly rolling backups.
    ✅ UPDATED: Compatible with new artifact structure.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Active model path
    active_path = os.path.join(MODEL_DIR, f"{symbol}_{mode}_xgb.pkl")
    joblib.dump(artifact, active_path)
    print(f"✅ Active {mode} model saved: {active_path}")

    # Monthly backup
    month_tag = datetime.now().strftime("%Y-%m")
    backup_path = os.path.join(MODEL_DIR, f"{symbol}_{mode}_xgb_{month_tag}.pkl")

    if not os.path.exists(backup_path):
        shutil.copy2(active_path, backup_path)
        print(f"📦 Monthly backup created: {backup_path}")
    else:
        print(f"ℹ️  Monthly backup for {symbol} ({mode}) in {month_tag} already exists — skipping.")

    # Cleanup old backups (per symbol+mode)
    cleanup_old_backups(symbol, mode)


def cleanup_old_backups(symbol: str, mode: str):
    """Remove old backups keeping only MAX_BACKUPS most recent."""
    pattern = os.path.join(MODEL_DIR, f"{symbol}_{mode}_xgb_*.pkl")
    backups = sorted(glob.glob(pattern))

    if len(backups) > MAX_BACKUPS:
        to_delete = backups[:-MAX_BACKUPS]
        for old in to_delete:
            try:
                os.remove(old)
                print(f"🗑️  Removed old backup: {os.path.basename(old)}")
            except Exception as e:
                print(f"[WARN] Could not remove backup {old}: {e}")


# ---------------------------------------------------------
# ✅ NEW: Rebuild features for SHAP (compatible with new pipeline)
# ---------------------------------------------------------

def rebuild_features_for_shap(df, symbol: str, mode: str):
    """
    Rebuild feature matrix matching the training pipeline.
    ✅ UPDATED: Works with new enhanced indicators.
    """
    from indicators import add_indicators
    
    # Add enhanced indicators (match training)
    try:
        df = add_indicators(df)
    except Exception as e:
        print(f"⚠️  Enhanced indicators failed: {e}, using basic features")
    
    # Build features based on mode
    if mode == "daily":
        from features import build_daily_features
        df_feat = build_daily_features(df)
    else:  # intraday modes
        from features import build_intraday_features
        df_feat = build_intraday_features(df)
        
        # Apply regime filtering
        if mode == "intraday_mr":
            df_feat = _filter_mr_regime(df_feat)
        elif mode == "intraday_mom":
            df_feat = _filter_mom_regime(df_feat)
    
    # Create target (binary for SHAP compatibility)
    df_feat["target"] = (df_feat["Close"].shift(-1) > df_feat["Close"]).astype(int)
    df_feat = df_feat.dropna()
    
    # Prepare X, y
    exclude_cols = ["target", "Close", "Open", "High", "Low", "Volume", 
                   "forward_return", "target_3class", "Target_multiclass"]
    X = df_feat.drop(columns=[c for c in exclude_cols if c in df_feat.columns], errors='ignore')
    y = df_feat["target"]
    
    return X, y


# ---------------------------------------------------------
# Train daily model with SHAP
# ---------------------------------------------------------

def train_daily_model_with_shap(sym: str):
    """
    Train daily model with optional SHAP feature selection.
    ✅ UPDATED: Uses new train_model() with Week 1 optimizations.
    """
    print(f"\n{'='*60}")
    print(f"🔄 Training {sym} - DAILY MODEL")
    print(f"{'='*60}")

    # ✅ UPDATED: Use new data fetching
    df_daily = fetch_historical_data(sym, period=DAILY_PERIOD, interval=DAILY_INTERVAL)

    if df_daily is None or df_daily.empty:
        print(f"[ERROR] No daily data for {sym}. Skipping.")
        return

    try:
        # ✅ UPDATED: Train with new pipeline
        print(f"🔨 Training model with Week 1 optimizations...")
        artifact = train_model(
            df_daily, 
            symbol=sym, 
            mode="daily",
            use_multiclass=USE_MULTICLASS  # ✅ NEW: Support multi-class
        )

        # ✅ OPTIONAL: SHAP feature selection (if available)
        if SHAP_AVAILABLE and SHAP_TOP_N:
            print(f"\n🔍 Running SHAP feature selection (top {SHAP_TOP_N})...")
            
            try:
                # Get model for SHAP (use base_model if calibrated)
                shap_model = artifact.get("base_model", artifact["model"])
                
                # Rebuild features
                X, y = rebuild_features_for_shap(df_daily, sym, "daily")
                
                # Split (match training)
                split_idx = int(len(X) * 0.8)
                X_train = X.iloc[:split_idx]
                X_test = X.iloc[split_idx:]
                
                # Run SHAP
                top_features, shap_vals, X_test_reduced = select_features_with_shap(
                    model=shap_model,
                    X_train=X_train,
                    X_test=X_test,
                    top_n=None,
                    plot=True,
                    symbol=sym,
                    mode="daily"
                )
                
                # Add SHAP metadata to artifact
                artifact["top_features"] = top_features
                artifact["feature_selection_method"] = "shap"
                artifact["shap_timestamp"] = datetime.now().isoformat()
                
                print(f"✅ SHAP analysis complete: {len(top_features)} top features")
                
            except Exception as e:
                print(f"⚠️  SHAP feature selection failed: {e}")
                print(f"⚠️  Proceeding without SHAP (model still trained)")

        # Save with backup
        save_model_with_backup(artifact, symbol=sym, mode="daily")
        
        # Print summary
        print_model_summary(artifact, sym, "daily")

    except Exception as e:
        print(f"[ERROR] Failed to train daily model for {sym}: {e}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------
# Train intraday models with SHAP
# ---------------------------------------------------------

def train_intraday_models_with_shap(sym: str):
    """
    Train intraday MR and MOM models with optional SHAP.
    ✅ UPDATED: Uses new train_model() with Week 1 optimizations.
    """
    print(f"\n{'='*60}")
    print(f"🔄 Training {sym} - INTRADAY MODELS")
    print(f"{'='*60}")

    # ✅ UPDATED: Use new data fetching
    df_intra = fetch_intraday_history(
        sym,
        lookback_minutes=INTRADAY_LOOKBACK,
        interval=INTRADAY_INTERVAL
    )

    if df_intra is None or df_intra.empty:
        print(f"[ERROR] No intraday data for {sym}. Skipping.")
        return
    


    # Train both intraday models
    for mode in ["intraday_mr", "intraday_mom"]:
        try:
            print(f"\n🔧 Training {sym} {mode.upper()}...")

            # ✅ UPDATED: Train with new pipeline
            artifact = train_model(
                df_intra, 
                symbol=sym, 
                mode=mode,
                use_multiclass=USE_MULTICLASS  # ✅ NEW: Support multi-class
            )

            # ✅ OPTIONAL: SHAP feature selection
            if SHAP_AVAILABLE and SHAP_TOP_N:
                print(f"🔍 Running SHAP feature selection...")
                
                try:
                    shap_model = artifact.get("base_model", artifact["model"])
                    X, y = rebuild_features_for_shap(df_intra, sym, mode)
                    split_idx = int(len(X) * 0.8)
                    X_train = X.iloc[:split_idx]
                    X_test = X.iloc[split_idx:]
                    top_features, shap_vals, X_test_reduced = select_features_with_shap(
                        model=shap_model,
                        X_train=X_train,
                        X_test=X_test,
                        top_n=None,
                        plot=True,
                        symbol=sym,
                        mode=mode
                    )
                    
                    artifact["top_features"] = top_features
                    artifact["feature_selection_method"] = "shap"
                    artifact["shap_timestamp"] = datetime.now().isoformat()
                    
                    print(f"✅ SHAP analysis complete: {len(top_features)} top features")
                    
                except Exception as e:
                    print(f"⚠️  SHAP feature selection failed: {e}")

            # Save with backup
            save_model_with_backup(artifact, symbol=sym, mode=mode)
            
            # Print summary
            print_model_summary(artifact, sym, mode)

        except ValueError as e:
            # ✅ GRACEFUL: Handle insufficient data
            if "Insufficient class diversity" in str(e):
                print(f"⚠️  Skipped {sym}/{mode}: Insufficient data after regime filtering")
            else:
                print(f"[ERROR] Failed to train {mode} for {sym}: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"[ERROR] Failed to train {mode} for {sym}: {e}")
            import traceback
            traceback.print_exc()


# ---------------------------------------------------------
# ✅ NEW: Print model summary
# ---------------------------------------------------------

def print_model_summary(artifact: dict, symbol: str, mode: str):
    """Print training summary."""
    metrics = artifact.get('metrics', {})
    
    print(f"\n{'─'*60}")
    print(f"📊 {symbol} - {mode.upper()}")
    print(f"{'─'*60}")
    print(f"  Type:       {artifact.get('target_type', 'binary')}")
    print(f"  Classes:    {artifact.get('num_classes', 2)}")
    print(f"  Calibrated: {artifact.get('calibrated', False)}")
    print(f"  Features:   {metrics.get('num_features', len(artifact.get('features', [])))}")
    
    if 'accuracy' in metrics and metrics['accuracy']:
        print(f"  Accuracy:   {metrics['accuracy']:.3f}")
    if 'logloss' in metrics and metrics['logloss']:
        print(f"  LogLoss:    {metrics['logloss']:.3f}")
    
    if artifact.get('top_features'):
        print(f"  SHAP:       {len(artifact['top_features'])} top features selected")
    
    print(f"{'─'*60}")


# ---------------------------------------------------------
# Regime filters (match model_xgb.py)
# ---------------------------------------------------------

def _filter_mr_regime(df_feat):
    """
    Filter for mean-reversion regime (low momentum, low vol).
    ✅ UPDATED: Match model_xgb.py percentile thresholds.
    """
    df_feat = df_feat.copy()
    df_feat["ret_12"] = df_feat["Close"].pct_change(12)
    df_feat["mom_12_abs"] = df_feat["ret_12"].abs()
    df_feat["vol_12"] = df_feat["Close"].pct_change().rolling(12).std()

    # Use 40th percentile (less aggressive, more data)
    mom_p40 = df_feat["mom_12_abs"].quantile(0.40)
    vol_p40 = df_feat["vol_12"].quantile(0.40)
    
    mask = (df_feat["mom_12_abs"] < mom_p40) & (df_feat["vol_12"] < vol_p40)
    
    filtered = df_feat[mask].drop(columns=["ret_12", "mom_12_abs", "vol_12"], errors='ignore')
    
    # Safety check
    if len(filtered) < 20:
        print(f"⚠️  MR filter too aggressive ({len(filtered)} rows), using 50th percentile")
        mom_p50 = df_feat["mom_12_abs"].quantile(0.50)
        vol_p50 = df_feat["vol_12"].quantile(0.50)
        mask = (df_feat["mom_12_abs"] < mom_p50) & (df_feat["vol_12"] < vol_p50)
        filtered = df_feat[mask].drop(columns=["ret_12", "mom_12_abs", "vol_12"], errors='ignore')
    
    return filtered


def _filter_mom_regime(df_feat):
    """
    Filter for momentum regime (high momentum OR high vol).
    ✅ UPDATED: Match model_xgb.py percentile thresholds.
    """
    df_feat = df_feat.copy()
    df_feat["ret_12"] = df_feat["Close"].pct_change(12)
    df_feat["mom_12_abs"] = df_feat["ret_12"].abs()
    df_feat["vol_12"] = df_feat["Close"].pct_change().rolling(12).std()


    mom_p60 = df_feat["mom_12_abs"].quantile(0.60)
    vol_p60 = df_feat["vol_12"].quantile(0.60)
    

    mask = (df_feat["mom_12_abs"] >= mom_p60) | (df_feat["vol_12"] >= vol_p60)
    return df_feat[mask].drop(columns=["ret_12", "mom_12_abs", "vol_12"], errors='ignore')


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    symbols = TRAIN_SYMBOLS if isinstance(TRAIN_SYMBOLS, list) else [TRAIN_SYMBOLS]

    print("\n" + "="*80)
    print("MODEL RETRAINING (Legacy SHAP-compatible mode)")
    print("="*80)
    print(f"\n💡 NOTE: This script uses Week 1 optimizations but maintains")
    print(f"   backward compatibility with SHAP feature selection.")
    print(f"\n   Consider using: python apply_week1_optimizations.py")
    print(f"   for the full feature set without SHAP overhead.\n")
    
    print(f"📌 Symbols: {symbols}")
    print(f"📂 MODEL_DIR: {os.path.abspath(MODEL_DIR)}")
    print(f"🔬 SHAP: {'Enabled' if SHAP_AVAILABLE else 'Disabled (not installed)'}")
    print(f"🎯 Target: {'Multi-class (5 classes)' if USE_MULTICLASS else 'Binary (2 classes)'}")
    print(f"🕒 {datetime.now().isoformat()}\n")

    for sym in symbols:
        # Daily model
        train_daily_model_with_shap(sym)

        # Intraday models (MR + MOM)
        train_intraday_models_with_shap(sym)

    print("\n" + "="*80)
    print("🎉 All retraining tasks complete.")
    print(f"📂 Models saved to: {os.path.abspath(MODEL_DIR)}")
    print("="*80)


if __name__ == "__main__":
    main()
