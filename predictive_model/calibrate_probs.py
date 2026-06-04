#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
import joblib  # pip install joblib if needed


LOGS_DIR = Path("logs")
MODELS_DIR = Path("models") / "calibrators"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def make_splits(df_all, mode, split_ratio=0.8, min_eval=20):
    """Filter by mode, keep labeled rows, sort by time, and split by time."""
    df = df_all[df_all["mode"] == mode].copy()
    df = df[df["actual_outcome"].isin([0, 1])].copy()
    if df.empty:
        return None, None

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")

    n = len(df)
    split_idx = int(split_ratio * n)
    df_calib = df.iloc[:split_idx].copy()
    df_eval = df.iloc[split_idx:].copy()

    # Skip if eval set is too small
    if len(df_eval) < min_eval or len(df_calib) == 0:
        return None, None

    return df_calib, df_eval


def fit_platt(df_calib):
    X = df_calib["predicted_prob"].values.reshape(-1, 1)
    y = df_calib["actual_outcome"].astype(int).values

    model = LogisticRegression(solver="lbfgs")
    model.fit(X, y)
    return model


def evaluate_calibration(df_eval, platt_model):
    X = df_eval["predicted_prob"].values.reshape(-1, 1)
    y = df_eval["actual_outcome"].astype(int).values

    p_raw = df_eval["predicted_prob"].values
    p_cal = platt_model.predict_proba(X)[:, 1]

    raw_brier = brier_score_loss(y, p_raw)
    cal_brier = brier_score_loss(y, p_cal)
    raw_logloss = log_loss(y, p_raw, labels=[0, 1])
    cal_logloss = log_loss(y, p_cal, labels=[0, 1])

    return raw_brier, cal_brier, raw_logloss, cal_logloss


def main():
    results = defaultdict(list)

    for csv_path in sorted(LOGS_DIR.glob("predictions_*_old.csv")):
        symbol = csv_path.stem.replace("predictions_", "")
        print(f"\n=========== Symbol: {symbol} ===========")

        df_all = pd.read_csv(csv_path)

        # Discover modes present in this file
        if "mode" not in df_all.columns:
            print("  [WARN] No 'mode' column, skipping file")
            continue

        modes = sorted(df_all["mode"].dropna().unique())

        for mode in modes:
            df_calib, df_eval = make_splits(df_all, mode)
            if df_calib is None:
                print(f"  [SKIP] {mode}: not enough data for calibration/eval")
                continue

            platt = fit_platt(df_calib)
            raw_brier, cal_brier, raw_logloss, cal_logloss = evaluate_calibration(df_eval, platt)

            label = f"{symbol} {mode}"
            print(f"\n  --- {label} ---")
            print(f"  Rows total: {len(df_calib) + len(df_eval)} "
                f"(calib: {len(df_calib)}, eval: {len(df_eval)})")
            print(f"  Raw Brier: {raw_brier:.6f}")
            print(f"  Cal  Brier: {cal_brier:.6f}")
            print(f"  Raw logloss: {raw_logloss:.6f}")
            print(f"  Cal  logloss: {cal_logloss:.6f}")

            # Decide if calibration is “good”
            improved = (cal_brier <= raw_brier) and (cal_logloss <= raw_logloss)
            print(f"  Improvement: {'YES' if improved else 'NO'}")

            if improved:
                fname = f"{symbol}_{mode}_platt.joblib"
                out_path = MODELS_DIR / fname
                joblib.dump(platt, out_path)
                print(f"  [SAVED] {out_path}")
            else:
                print("  [SKIP SAVE] Calibration hurt metrics; not saving calibrator.")

            results[symbol].append(
                (mode, raw_brier, cal_brier, raw_logloss, cal_logloss)
            )

    # Optional: print a compact summary at the end
    print("\n===== Summary =====")
    for symbol, entries in results.items():
        print(f"\nSymbol: {symbol}")
        for mode, rb, cb, rl, cl in entries:
            print(
                f"  {mode:15s} | "
                f"Brier {rb:.3f} -> {cb:.3f} | "
                f"logloss {rl:.3f} -> {cl:.3f}"
            )


if __name__ == "__main__":
    main()