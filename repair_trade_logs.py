import os
import glob
import shutil
from datetime import datetime

import pandas as pd


def repair_trade_csv(path: str, make_backup: bool = True) -> bool:
    """
    Repair an old trades_<symbol>.csv:
      - ensure shares_before/shares_after exist
      - fix qty when it's 0/NaN using abs(delta_shares)
    Returns True if file was changed.
    """
    df = pd.read_csv(path)
    if df.empty:
        return False

    # Normalize required columns
    if "timestamp" not in df.columns:
        print(f"[SKIP] No timestamp in {path}")
        return False

    # Parse timestamp safely
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Ensure numeric fields
    for col in ["qty", "price", "cash", "shares", "value"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # If already repaired and looks good, still sanity-check qty==0 rows
    changed = False

    # Create shares_before/after from existing "shares" (which is post-trade shares in your logger)
    if "shares_after" not in df.columns:
        df["shares_after"] = df["shares"]
        changed = True

    if "shares_before" not in df.columns:
        df["shares_before"] = df["shares_after"].shift(1).fillna(0.0)
        changed = True

    # Keep original qty for audit
    if "qty_raw" not in df.columns:
        df["qty_raw"] = df["qty"]
        changed = True

    # Recompute qty from delta shares when qty is invalid
    delta = (df["shares_after"] - df["shares_before"]).abs()

    # Define "bad qty": NaN or <= 0 but a position change occurred
    bad_qty_mask = (df["qty"].isna() | (df["qty"] <= 0)) & (delta > 0)

    if bad_qty_mask.any():
        df.loc[bad_qty_mask, "qty"] = delta[bad_qty_mask]
        changed = True

    # Optional: mark repaired rows
    if "repaired" not in df.columns:
        df["repaired"] = False
        changed = True
    df.loc[bad_qty_mask, "repaired"] = True

    # Also fix any accidental negative qty
    neg_mask = df["qty"] < 0
    if neg_mask.any():
        df.loc[neg_mask, "qty"] = df.loc[neg_mask, "qty"].abs()
        df.loc[neg_mask, "repaired"] = True
        changed = True

    # If no changes needed, exit
    if not changed:
        return False

    # Backup then overwrite
    if make_backup:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{path}.bak_{stamp}"
        shutil.copy2(path, backup_path)
        print(f"[BACKUP] {backup_path}")

    df.to_csv(path, index=False)
    print(f"[REPAIRED] {path}  (fixed_rows={int(bad_qty_mask.sum())})")
    return True


def repair_all_trade_logs(trade_dir: str):
    paths = sorted(glob.glob(os.path.join(trade_dir, "trades_*.csv")))
    if not paths:
        print(f"[INFO] No trades_*.csv found in {trade_dir}")
        return

    changed_any = False
    for p in paths:
        try:
            changed = repair_trade_csv(p, make_backup=True)
            changed_any = changed_any or changed
        except Exception as e:
            print(f"[ERROR] Failed repairing {p}: {e}")

    if not changed_any:
        print("[DONE] Nothing needed repair.")
    else:
        print("[DONE] Repair complete.")


if __name__ == "__main__":
    # 🔧 Set this to wherever get_trade_log_file() writes logs
    TRADE_DIR = "./data"  # <- change if your logs live elsewhere
    repair_all_trade_logs(TRADE_DIR)