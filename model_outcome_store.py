"""Durable, environment-scoped storage for resolved model predictions."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from config import DATA_DIR


OUTCOME_COLUMNS = [
    "timestamp",
    "symbol",
    "mode",
    "predicted_prob",
    "price",
    "actual_outcome",
    "actual_price",
    "return_pct",
    "outcome_horizon",
    "target_horizon",
    "actual_movement",
    "actual_direction",
]
OUTCOME_KEY = ["timestamp", "symbol", "mode", "outcome_horizon"]


def outcome_history_path() -> Path:
    configured = os.getenv("MODEL_OUTCOME_HISTORY_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(DATA_DIR).resolve() / "model_outcomes.csv"


def _normalized_resolved_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=OUTCOME_COLUMNS)

    result = frame.copy()
    for column in OUTCOME_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA

    result["timestamp"] = pd.to_datetime(
        result["timestamp"], utc=True, errors="coerce"
    )
    result["actual_outcome"] = pd.to_numeric(
        result["actual_outcome"], errors="coerce"
    )
    result = result.dropna(subset=["timestamp", "symbol", "mode", "actual_outcome"])
    result["symbol"] = result["symbol"].astype(str).str.upper().str.strip()
    result["mode"] = result["mode"].astype(str).str.lower().str.strip()
    result = result.loc[result["actual_outcome"].isin([0, 1])]
    result["timestamp"] = result["timestamp"].map(lambda value: value.isoformat())
    result["outcome_horizon"] = result["outcome_horizon"].fillna("").astype(str)
    return result[OUTCOME_COLUMNS]


def upsert_resolved_outcomes(
    frame: pd.DataFrame,
    path: Path = None,
) -> int:
    """Atomically insert resolved rows and return the number of new keys."""
    path = Path(path or outcome_history_path())
    incoming = _normalized_resolved_rows(frame)
    if incoming.empty:
        return 0

    if path.exists():
        existing = _normalized_resolved_rows(pd.read_csv(path))
    else:
        existing = pd.DataFrame(columns=OUTCOME_COLUMNS)

    previous_keys = set(map(tuple, existing[OUTCOME_KEY].astype(str).to_numpy()))
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.drop_duplicates(subset=OUTCOME_KEY, keep="last")
    combined = combined.sort_values(["timestamp", "symbol", "mode"])

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    os.replace(temporary, path)

    current_keys = set(map(tuple, combined[OUTCOME_KEY].astype(str).to_numpy()))
    return len(current_keys - previous_keys)


def load_outcome_history(
    *,
    symbol: str = None,
    mode: str = None,
    path: Path = None,
) -> pd.DataFrame:
    path = Path(path or outcome_history_path())
    if not path.exists():
        return pd.DataFrame(columns=OUTCOME_COLUMNS)

    frame = _normalized_resolved_rows(pd.read_csv(path))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if symbol is not None:
        frame = frame.loc[frame["symbol"] == str(symbol).upper()]
    if mode is not None:
        frame = frame.loc[frame["mode"] == str(mode).lower()]
    return frame.sort_values("timestamp").reset_index(drop=True)
