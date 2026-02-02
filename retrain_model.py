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
# Train daily model with feature selection
# ---------------------------------------------------------
def train_daily_model_with_shap(sym: str):
    """Train daily model with SHAP feature selection."""
    print(f"\n{'='*60}")
    print(f"🔄 Training {sym} - DAILY MODEL")
    print(f"{'='*60}")

    df_daily = fetch_historical_data(sym, years=LOOKBACK_YEARS, interval=DAILY_INTERVAL)

    if df_daily is None or df_daily.empty:
        print(f"[ERROR] No daily data for {sym}. Skipping.")
        return

    try:
        df_daily = _clean_columns(df_daily)

        # Train initial model
        artifact_daily = train_model(df_daily, symbol=sym, mode="daily")

        # Extract trained model
        model = artifact_daily["model"]

        # Rebuild features for SHAP (match training pipeline)
        from features import build_daily_features
        from target_labels import create_target_label

        df_feat = build_daily_features(df_daily)
        df_feat = create_target_label(df_feat, mode="daily")
        df_feat = df_feat.dropna(subset=["target"])

        X = df_feat.drop(columns=["target", "forward_return", "target_3class"], errors='ignore')
        y = df_feat["target"]

        # Split (match model_xgb training split)
        train_end = int(len(X) * 0.6)
        val_end = int(len(X) * 0.8)

        X_train = X.iloc[:train_end]
        X_test = X.iloc[val_end:]
        y_train = y.iloc[:train_end]
        y_test = y.iloc[val_end:]

        # ✅ FIXED: Pass symbol and mode to SHAP
        top_features, shap_vals, X_test_reduced = select_features_with_shap(
            model=model,
            X_train=X_train,
            X_test=X_test,
            top_n=30,
            plot=True,
            symbol=sym,      # ✅ From config
            mode="daily"      # ✅ Explicit
        )

        # Save artifact with top features
        artifact_daily["top_features"] = top_features
        artifact_daily["feature_selection_method"] = "shap"
        artifact_daily["shap_timestamp"] = datetime.now().isoformat()

        save_model_with_backup(artifact_daily, symbol=sym, mode="daily")

    except Exception as e:
        print(f"[ERROR] Failed to train daily model for {sym}: {e}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------
# Train intraday models with feature selection
# ---------------------------------------------------------
def train_intraday_models_with_shap(sym: str):
    """Train intraday MR and MOM models with SHAP."""
    print(f"\n{'='*60}")
    print(f"🔄 Training {sym} - INTRADAY MODELS")
    print(f"{'='*60}")

    df_intra = fetch_historical_data(
        sym,
        period=f"{INTRADAY_LOOKBACK_DAYS}d",
        interval=INTRADAY_INTERVAL
    )

    if df_intra is None or df_intra.empty:
        print(f"[ERROR] No intraday data for {sym}. Skipping.")
        return

    df_intra = _clean_columns(df_intra)

    # Train both intraday models
    for mode in ["intraday_mr", "intraday_mom"]:
        try:
            print(f"\n🔧 Training {sym} {mode.upper()}...")

            # Train model
            artifact = train_model(df_intra, symbol=sym, mode=mode)
            model = artifact["model"]

            # Rebuild features for SHAP
            from features import build_intraday_features
            from target_labels import create_target_label

            df_feat = build_intraday_features(df_intra)
            df_feat = create_target_label(df_feat, mode="intraday")
            df_feat = df_feat.dropna(subset=["target"])

            # Apply regime filtering (match training)
            if mode == "intraday_mr":
                df_feat = _filter_mr_regime(df_feat)
            elif mode == "intraday_mom":
                df_feat = _filter_mom_regime(df_feat)

            X = df_feat.drop(columns=["target", "forward_return", "target_3class"], errors='ignore')
            y = df_feat["target"]

            # Split
            split_idx = int(len(X) * 0.8)
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]

            # ✅ FIXED: Pass symbol and mode to SHAP
            top_features, shap_vals, X_test_reduced = select_features_with_shap(
                model=model,
                X_train=X_train,
                X_test=X_test,
                top_n=30,
                plot=True,
                symbol=sym,      # ✅ From config
                mode=mode         # ✅ intraday_mr or intraday_mom
            )

            # Save artifact with top features
            artifact["top_features"] = top_features
            artifact["feature_selection_method"] = "shap"
            artifact["shap_timestamp"] = datetime.now().isoformat()

            save_model_with_backup(artifact, symbol=sym, mode=mode)

        except Exception as e:
            print(f"[ERROR] Failed to train {mode} for {sym}: {e}")
            import traceback
            traceback.print_exc()


# ---------------------------------------------------------
# Regime filters (match model_xgb.py logic)
# ---------------------------------------------------------
def _filter_mr_regime(df_feat):
    """Filter for mean-reversion regime (low momentum, low vol)."""
    df_feat = df_feat.copy()
    df_feat["ret_12"] = df_feat["Close"].pct_change(12)
    df_feat["mom_12_abs"] = df_feat["ret_12"].abs()
    df_feat["vol_12"] = df_feat["Close"].pct_change().rolling(12).std()

    # MR: low momentum AND low vol
    mask = (df_feat["mom_12_abs"] < 0.010) & (df_feat["vol_12"] < 0.010)
    return df_feat[mask].drop(columns=["ret_12", "mom_12_abs", "vol_12"], errors='ignore')


def _filter_mom_regime(df_feat):
    """Filter for momentum regime (high momentum OR high vol)."""
    df_feat = df_feat.copy()
    df_feat["ret_12"] = df_feat["Close"].pct_change(12)
    df_feat["mom_12_abs"] = df_feat["ret_12"].abs()
    df_feat["vol_12"] = df_feat["Close"].pct_change().rolling(12).std()

    # MOM: high momentum OR high vol
    mask = (df_feat["mom_12_abs"] >= 0.010) | (df_feat["vol_12"] >= 0.010)
    return df_feat[mask].drop(columns=["ret_12", "mom_12_abs", "vol_12"], errors='ignore')


# ---------------------------------------------------------
# Main multi-symbol retraining loop
# ---------------------------------------------------------
def main():
    symbols = TRAIN_SYMBOLS if isinstance(TRAIN_SYMBOLS, list) else [TRAIN_SYMBOLS]

    print("\n🔄 Starting model retraining with SHAP feature selection...")
    print(f"📌 Symbols: {symbols}")
    print(f"📂 MODEL_DIR: {os.path.abspath(MODEL_DIR)}")
    print(f"🕒 {datetime.now().isoformat()}\n")

    for sym in symbols:
        # Daily model
        train_daily_model_with_shap(sym)

        # Intraday models (MR + MOM)
        train_intraday_models_with_shap(sym)

    print("\n" + "="*60)
    print("🎉 All retraining tasks complete.")
    print(f"📂 Models saved to: {os.path.abspath(MODEL_DIR)}")
    print("="*60)


if __name__ == "__main__":
    main()
