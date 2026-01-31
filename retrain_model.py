# retrain_model.py
import os
import glob
import shutil
from datetime import datetime

import joblib

from data_loader import fetch_historical_data
from model_xgb import train_model, MODEL_DIR
from features import _clean_columns
from config import TRAIN_SYMBOLS
from feature_selection import select_features_with_shap, retrain_with_selected_features

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------
MAX_BACKUPS = 6  # keep only last 6 backups
LOOKBACK_YEARS = 2
DAILY_INTERVAL = "1d"

INTRADAY_LOOKBACK_DAYS = 60
INTRADAY_INTERVAL = "15m"


# ---------------------------------------------------------
# Save model artifact + rolling backups
# ---------------------------------------------------------
def save_model_with_backup(artifact, symbol, mode: str = "daily"):
    """
    Save active model artifact and maintain monthly rolling backups.
    Artifact should be the dict returned by train_model().
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
        print(f"ℹ️ Monthly backup for {symbol} ({mode}) in {month_tag} already exists — skipping.")

    # Cleanup old backups (per symbol+mode)
    pattern = os.path.join(MODEL_DIR, f"{symbol}_{mode}_xgb_*.pkl")
    backups = sorted(glob.glob(pattern))

    if len(backups) > MAX_BACKUPS:
        to_delete = backups[:-MAX_BACKUPS]
        for old in to_delete:
            try:
                os.remove(old)
                print(f"🗑️ Removed old backup for {symbol} ({mode}): {old}")
            except Exception as e:
                print(f"[WARN] Could not remove backup {old}: {e}")


# ---------------------------------------------------------
# Main multi-symbol retraining loop
# ---------------------------------------------------------
def main():
    symbols = TRAIN_SYMBOLS if isinstance(TRAIN_SYMBOLS, list) else [TRAIN_SYMBOLS]
    
    print("\n🔄 Starting model retraining with feature selection...\n")
    
    for sym in symbols:
        # Train initial model (as before)
        print(f"\n{'='*60}")
        print(f"🔄 Training {sym} - DAILY MODEL")
        print(f"{'='*60}")
        
        df_daily = fetch_historical_data(sym, years=LOOKBACK_YEARS, interval=DAILY_INTERVAL)
        
        if df_daily is None or df_daily.empty:
            print(f"[ERROR] No daily data for {sym}. Skipping.")
            continue
        
        try:
            df_daily = _clean_columns(df_daily)
            
            # Train with validation
            artifact_daily = train_model(df_daily, symbol=sym, mode="daily")
            
            # Extract trained model and data
            model = artifact_daily["model"]
            features = artifact_daily["features"]
            split_info = artifact_daily.get("split_info", {})
            
            # Reconstruct X_train, X_test for SHAP
            # (You'll need to save these in train_model_with_validation)
            # For now, we'll retrain quickly to get them:
            
            # Rebuild features
            from model_xgb import build_daily_features
            from target_labels import create_target_label
            
            df_feat = build_daily_features(df_daily)
            df_feat = create_target_label(df_feat, mode="daily")
            df_feat = df_feat.dropna(subset=["target"])
            
            X = df_feat.drop(columns=["target", "forward_return", "target_3class"])
            y = df_feat["target"]
            
            # Split (same as training)
            train_end = int(len(X) * 0.6)
            val_end = int(len(X) * 0.8)
            
            X_train = X.iloc[:train_end]
            X_test = X.iloc[val_end:]
            y_test = y.iloc[val_end:]
            
            # ✅ SHAP Feature Selection
            top_features, shap_vals, X_test_reduced = select_features_with_shap(
                model=model,
                X_train=X_train,
                X_test=X_test,
                top_n=30,
                plot=True
            )
            
            # ✅ Optional: Retrain with reduced features
            # (Skip if performance doesn't improve significantly)
            
            # Save artifact with top features
            artifact_daily["top_features"] = top_features
            artifact_daily["feature_selection_method"] = "shap"
            
            save_model_with_backup(artifact_daily, symbol=sym, mode="daily")
            
        except Exception as e:
            print(f"[ERROR] Failed to train daily model for {sym}: {e}")
            import traceback
            traceback.print_exc()
        
        # Repeat for intraday models...
        
    print("\n🎉 All retraining tasks complete.\n")


if __name__ == "__main__":
    main()