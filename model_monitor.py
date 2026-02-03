#!/usr/bin/env python
"""
Model performance monitoring utilities.
Tracks prediction logs and evaluates model accuracy over time.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from config import MODEL_DIR



def log_prediction(
    symbol: str,
    mode: str,
    predicted_prob: float,
    actual_outcome: Optional[int] = None,
    price: Optional[float] = None,
    logs_dir: str = "logs"
):
    """
    Log a single prediction to CSV for later evaluation.

    Args:
        symbol: Stock symbol (e.g., 'NVDA')
        mode: Model mode ('daily', 'intraday_mr', 'intraday_mom')
        predicted_prob: Model's predicted probability (0.0-1.0)
        actual_outcome: Actual result (0=down, 1=up) - can be None initially
        price: Current price at prediction time
        logs_dir: Directory to store logs

    Format: predictions_{SYMBOL}_{MODE}.csv
    Columns: timestamp, symbol, mode, predicted_prob, actual_outcome, price
    """
    os.makedirs(logs_dir, exist_ok=True)

    # File per symbol-mode combination
    filename = f"predictions_{symbol.upper()}_{mode}.csv"
    path = os.path.join(logs_dir, filename)

    # Create row
    row = {
        'timestamp': pd.Timestamp.utcnow().isoformat(),
        'symbol': symbol.upper(),
        'mode': mode,
        'predicted_prob': float(predicted_prob),
        'actual_outcome': actual_outcome if actual_outcome is not None else np.nan,
        'price': float(price) if price is not None else np.nan
    }

    # Append to CSV
    df_row = pd.DataFrame([row])

    if os.path.exists(path):
        df_row.to_csv(path, mode='a', header=False, index=False)
    else:
        df_row.to_csv(path, mode='w', header=True, index=False)


def evaluate_predictions(
    symbol: str,
    mode: str,
    lookback_days: int = 7,
    logs_dir: str = "logs"
) -> Dict:
    """
    Evaluate model performance from logged predictions.

    Args:
        symbol: Stock symbol
        mode: Model mode
        lookback_days: Days of history to analyze
        logs_dir: Directory containing logs

    Returns:
        Dict with metrics (accuracy, brier_score, calibration_error, etc.)
    """
    filename = f"predictions_{symbol.upper()}_{mode}.csv"
    path = os.path.join(logs_dir, filename)

    if not os.path.exists(path):
        return {
            'sample_size': 0,
            'accuracy': 0.0,
            'brier_score': 1.0,
            'calibration_error': 0.0,
            'log_loss': 10.0
        }

    try:
        df = pd.read_csv(path)

        # ✅ Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            df = df.dropna(subset=['timestamp'])

        # Filter by lookback
        cutoff = pd.Timestamp.utcnow() - timedelta(days=lookback_days)
        df = df[df['timestamp'] >= cutoff].copy()

        # Remove rows without actual outcomes
        df = df.dropna(subset=['actual_outcome', 'predicted_prob'])

        if len(df) < 5:
            return {
                'sample_size': len(df),
                'accuracy': 0.0,
                'brier_score': 1.0,
                'calibration_error': 0.0,
                'log_loss': 10.0
            }

        y_true = df['actual_outcome'].astype(int).values
        y_pred_prob = df['predicted_prob'].values
        y_pred = (y_pred_prob >= 0.5).astype(int)

        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        brier = brier_score_loss(y_true, y_pred_prob)

        try:
            ll = log_loss(y_true, y_pred_prob, labels=[0, 1])
        except:
            ll = 10.0

        # Calibration error
        avg_pred = float(np.mean(y_pred_prob))
        actual_rate = float(np.mean(y_true))
        calib_error = abs(avg_pred - actual_rate)

        return {
            'sample_size': len(df),
            'accuracy': float(acc),
            'brier_score': float(brier),
            'calibration_error': float(calib_error),
            'log_loss': float(ll),
            'avg_predicted_prob': avg_pred,
            'actual_win_rate': actual_rate
        }

    except Exception as e:
        print(f"[ERROR] evaluate_predictions({symbol}, {mode}): {e}")
        return {
            'sample_size': 0,
            'accuracy': 0.0,
            'brier_score': 1.0,
            'calibration_error': 0.0,
            'log_loss': 10.0
        }


class ModelMonitor:
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = logs_dir
        os.makedirs(logs_dir, exist_ok=True)


    def _get_prediction_log_path(self, symbol: str, mode: str) -> str:
        """Get path to prediction log CSV."""
        return os.path.join(self.logs_dir, f"signals_{symbol.upper()}.csv")


    def _load_prediction_log(self, symbol: str, lookback_days: int = 7) -> pd.DataFrame:
        """
        Load prediction log with proper datetime handling.

        ✅ FIX: Convert timestamp column to datetime BEFORE any comparisons.
        """
        path = self._get_prediction_log_path(symbol, "")

        if not os.path.exists(path):
            return pd.DataFrame()

        try:
            df = pd.read_csv(path)

            # ✅ CRITICAL FIX: Convert timestamp to datetime IMMEDIATELY
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
                df = df.dropna(subset=['timestamp'])  # Remove invalid timestamps
            else:
                return pd.DataFrame()

            # Now safe to do datetime comparisons
            cutoff = pd.Timestamp.utcnow() - timedelta(days=lookback_days)
            df = df[df['timestamp'] >= cutoff].copy()

            return df.sort_values('timestamp')

        except Exception as e:
            print(f"[ERROR] _load_prediction_log({symbol}): {e}")
            return pd.DataFrame()


    def get_performance_metrics(
        self,
        symbol: str,
        mode: str,
        lookback_days: int = 7
    ) -> Dict:
        """
        Calculate model performance metrics from prediction logs.

        Args:
            symbol: Stock symbol
            mode: 'daily', 'intraday_mr', or 'intraday_mom'
            lookback_days: Days of history to analyze

        Returns:
            Dict with accuracy, calibration_error, brier_score, etc.
        """
        df = self._load_prediction_log(symbol, lookback_days)

        if df.empty or len(df) < 5:
            return {
                'sample_size': 0,
                'accuracy': 0.0,
                'calibration_error': 0.0,
                'brier_score': 1.0,
                'log_loss': 10.0,
                'avg_predicted_prob': 0.5,
                'actual_win_rate': 0.5
            }

        # Map mode to column
        mode_col_map = {
            'daily': 'dailyprob',
            'intraday_mr': 'intradayprob',  # Approximate
            'intraday_mom': 'intradayprob'  # Approximate
        }

        prob_col = mode_col_map.get(mode, 'finalprob')

        if prob_col not in df.columns or 'price' not in df.columns:
            return {
                'sample_size': 0,
                'accuracy': 0.0,
                'calibration_error': 0.0,
                'brier_score': 1.0,
                'log_loss': 10.0,
                'avg_predicted_prob': 0.5,
                'actual_win_rate': 0.5
            }

        # Calculate actual outcomes (next day's price movement)
        df = df.copy()
        df['actual_up'] = (df['price'].shift(-1) > df['price']).astype(int)
        df = df.dropna(subset=['actual_up', prob_col])

        if len(df) < 5:
            return {
                'sample_size': len(df),
                'accuracy': 0.0,
                'calibration_error': 0.0,
                'brier_score': 1.0,
                'log_loss': 10.0,
                'avg_predicted_prob': 0.5,
                'actual_win_rate': 0.5
            }

        y_true = df['actual_up'].values
        y_pred_prob = df[prob_col].values
        y_pred = (y_pred_prob >= 0.5).astype(int)

        # Metrics
        acc = accuracy_score(y_true, y_pred)
        brier = brier_score_loss(y_true, y_pred_prob)

        try:
            ll = log_loss(y_true, y_pred_prob, labels=[0, 1])
        except:
            ll = 10.0

        # Calibration error (expected vs actual)
        avg_pred = float(np.mean(y_pred_prob))
        actual_win_rate = float(np.mean(y_true))
        calib_error = abs(avg_pred - actual_win_rate)

        return {
            'sample_size': len(df),
            'accuracy': float(acc),
            'calibration_error': float(calib_error),
            'brier_score': float(brier),
            'log_loss': float(ll),
            'avg_predicted_prob': avg_pred,
            'actual_win_rate': actual_win_rate
        }


    def needs_retraining(
        self,
        symbol: str,
        mode: str,
        lookback_days: int = 7
    ) -> Tuple[bool, str]:
        """
        Determine if a model needs retraining based on performance.

        Returns:
            (needs_retrain: bool, reason: str)
        """
        metrics = self.get_performance_metrics(symbol, mode, lookback_days)

        if metrics['sample_size'] < 5:
            return False, "Insufficient data"

        # Thresholds
        if metrics['accuracy'] < 0.48:
            return True, f"Low accuracy: {metrics['accuracy']:.2%}"

        if metrics['calibration_error'] > 0.15:
            return True, f"Poor calibration: {metrics['calibration_error']:.2%}"

        if metrics['brier_score'] > 0.30:
            return True, f"High Brier score: {metrics['brier_score']:.4f}"

        return False, "Performing well"


    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Remove prediction logs older than specified days."""
        cutoff = pd.Timestamp.utcnow() - timedelta(days=days_to_keep)

        for filename in os.listdir(self.logs_dir):
            if not filename.startswith("signals_") or not filename.endswith(".csv"):
                continue

            path = os.path.join(self.logs_dir, filename)

            try:
                df = pd.read_csv(path)

                # ✅ FIX: Convert timestamp before comparison
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
                    df = df.dropna(subset=['timestamp'])

                    # Keep only recent data
                    df_clean = df[df['timestamp'] >= cutoff]

                    if len(df_clean) < len(df):
                        df_clean.to_csv(path, index=False)
                        print(f"✅ Cleaned {filename}: {len(df)} → {len(df_clean)} rows")

            except Exception as e:
                print(f"⚠️ Error cleaning {filename}: {e}")
