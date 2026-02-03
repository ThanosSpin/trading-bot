# model_monitor.py
"""
Model performance monitoring system.
Tracks predictions vs actual outcomes to detect model degradation.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import warnings

# Try to import MODEL_DIR from config/model_xgb
try:
    from model_xgb import MODEL_DIR
except ImportError:
    try:
        from config import MODEL_DIR
    except ImportError:
        MODEL_DIR = "models"


class ModelMonitor:
    """
    Tracks model predictions and compares with actual outcomes.

    Logs each prediction to CSV:
    - timestamp, symbol, mode, predicted_prob, actual_outcome, actual_return

    Alerts when model performance degrades below threshold.
    """

    def __init__(self, model_dir: str = None):
        """
        Initialize monitor.

        Args:
            model_dir: Directory to store prediction logs (default: models/)
        """
        self.model_dir = model_dir or MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)


    def log_prediction(
        self,
        symbol: str,
        mode: str,
        predicted_prob: float,
        price_at_prediction: float,
        timestamp: datetime = None
    ):
        """
        Log a prediction to track later.

        Args:
            symbol: Stock symbol (e.g., "NVDA")
            mode: Model mode (e.g., "intraday_mr", "intraday_mom", "daily")
            predicted_prob: Model's predicted probability (0-1)
            price_at_prediction: Stock price at prediction time
            timestamp: When prediction was made (default: now)
        """
        timestamp = timestamp or datetime.now()

        log_path = self._get_log_path(symbol, mode)

        row = {
            'timestamp': timestamp.isoformat(),
            'predicted_prob': round(predicted_prob, 4),
            'price_at_prediction': round(price_at_prediction, 2),
            'actual_outcome': None,  # Filled in later
            'actual_return': None,   # Filled in later
            'evaluated': False
        }

        # Append to CSV
        df = pd.DataFrame([row])
        df.to_csv(
            log_path,
            mode='a',
            header=not os.path.exists(log_path),
            index=False
        )

        print(f"[MONITOR] Logged prediction: {symbol}/{mode} prob={predicted_prob:.3f}")


    def evaluate_predictions(
        self,
        symbol: str,
        mode: str,
        current_price: float,
        lookback_hours: int = 24
    ) -> Dict[str, any]:
        """
        Evaluate past predictions by comparing with actual price movements.

        Args:
            symbol: Stock symbol
            mode: Model mode
            current_price: Current stock price
            lookback_hours: How far back to evaluate (default: 24h)

        Returns:
            {
                'evaluated_count': int,
                'accuracy': float,
                'brier_score': float,
                'calibration_error': float
            }
        """
        log_path = self._get_log_path(symbol, mode)

        if not os.path.exists(log_path):
            return {'evaluated_count': 0}

        df = pd.read_csv(log_path, parse_dates=['timestamp'])

        # Find unevaluated predictions older than evaluation window
        cutoff = datetime.now() - timedelta(hours=lookback_hours)

        if mode.startswith('intraday'):
            eval_window_hours = 2  # Intraday: evaluate after 2 hours
        else:
            eval_window_hours = 24  # Daily: evaluate next day

        eval_cutoff = datetime.now() - timedelta(hours=eval_window_hours)

        # Predictions ready to evaluate
        mask = (
            (df['evaluated'] == False) &
            (df['timestamp'] < eval_cutoff)
        )

        to_eval = df[mask].copy()

        if len(to_eval) == 0:
            return {'evaluated_count': 0}

        # Evaluate each prediction
        for idx in to_eval.index:
            pred_price = df.loc[idx, 'price_at_prediction']

            # Actual outcome: did price go up?
            actual_return = (current_price - pred_price) / pred_price
            actual_outcome = 1 if actual_return > 0 else 0

            df.loc[idx, 'actual_outcome'] = actual_outcome
            df.loc[idx, 'actual_return'] = round(actual_return, 5)
            df.loc[idx, 'evaluated'] = True

        # Save updated log
        df.to_csv(log_path, index=False)

        print(f"[MONITOR] Evaluated {len(to_eval)} predictions for {symbol}/{mode}")

        return {'evaluated_count': len(to_eval)}


    def get_performance_metrics(
        self,
        symbol: str,
        mode: str,
        lookback_days: int = 7
    ) -> Dict[str, float]:
        """
        Calculate model performance over recent period.

        Args:
            symbol: Stock symbol
            mode: Model mode
            lookback_days: Analysis window (default: 7 days)

        Returns:
            {
                'sample_size': int,
                'accuracy': float (0-1),
                'avg_predicted_prob': float,
                'actual_win_rate': float,
                'calibration_error': float (|predicted - actual|),
                'brier_score': float (0-1, lower is better),
                'log_loss': float
            }
        """
        log_path = self._get_log_path(symbol, mode)

        if not os.path.exists(log_path):
            return {'sample_size': 0}

        df = pd.read_csv(log_path, parse_dates=['timestamp'])

        # Filter to evaluated predictions in lookback window
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent = df[
            (df['evaluated'] == True) &
            (df['timestamp'] > cutoff)
        ].copy()

        if len(recent) < 5:  # Need minimum sample
            return {'sample_size': len(recent)}

        # Calculate metrics
        pred = recent['predicted_prob'].values
        actual = recent['actual_outcome'].values

        accuracy = (pred > 0.5) == actual
        accuracy = accuracy.mean()

        avg_prob = pred.mean()
        win_rate = actual.mean()
        calibration_error = abs(avg_prob - win_rate)

        # Brier score (mean squared error of probabilities)
        brier = np.mean((pred - actual) ** 2)

        # Log loss
        eps = 1e-15  # Avoid log(0)
        pred_clipped = np.clip(pred, eps, 1 - eps)
        log_loss = -np.mean(
            actual * np.log(pred_clipped) +
            (1 - actual) * np.log(1 - pred_clipped)
        )

        return {
            'sample_size': len(recent),
            'accuracy': round(accuracy, 4),
            'avg_predicted_prob': round(avg_prob, 4),
            'actual_win_rate': round(win_rate, 4),
            'calibration_error': round(calibration_error, 4),
            'brier_score': round(brier, 4),
            'log_loss': round(log_loss, 4)
        }


    def needs_retraining(
        self,
        symbol: str,
        mode: str,
        lookback_days: int = 7,
        min_samples: int = 20,
        calibration_threshold: float = 0.15,
        accuracy_threshold: float = 0.45
    ) -> Tuple[bool, str]:
        """
        Determine if model needs retraining based on performance.

        Args:
            symbol: Stock symbol
            mode: Model mode
            lookback_days: Analysis window
            min_samples: Minimum predictions needed to evaluate
            calibration_threshold: Alert if |predicted - actual| > this
            accuracy_threshold: Alert if accuracy < this

        Returns:
            (needs_retrain: bool, reason: str)
        """
        metrics = self.get_performance_metrics(symbol, mode, lookback_days)

        if metrics.get('sample_size', 0) < min_samples:
            return False, f"Insufficient data ({metrics.get('sample_size', 0)} < {min_samples})"

        # Check calibration error
        cal_error = metrics.get('calibration_error', 0)
        if cal_error > calibration_threshold:
            return True, f"Calibration error {cal_error:.2%} > {calibration_threshold:.2%}"

        # Check accuracy
        acc = metrics.get('accuracy', 0)
        if acc < accuracy_threshold:
            return True, f"Accuracy {acc:.2%} < {accuracy_threshold:.2%}"

        return False, "Performance OK"


    def print_performance_report(
        self,
        symbols: list,
        modes: list = None
    ):
        """
        Print performance report for all models.

        Args:
            symbols: List of symbols to check
            modes: List of modes (default: all modes)
        """
        if modes is None:
            modes = ['daily', 'intraday_mr', 'intraday_mom']

        print("\n" + "="*80)
        print("MODEL PERFORMANCE REPORT")
        print("="*80)

        for sym in symbols:
            for mode in modes:
                metrics = self.get_performance_metrics(sym, mode, lookback_days=7)

                if metrics.get('sample_size', 0) < 5:
                    continue

                needs_retrain, reason = self.needs_retraining(sym, mode)

                status = "⚠️ RETRAIN" if needs_retrain else "✅ OK"

                print(f"\n{status} {sym}/{mode}")
                print(f"  Samples: {metrics['sample_size']}")
                print(f"  Accuracy: {metrics['accuracy']:.2%}")
                print(f"  Calibration: predicted={metrics['avg_predicted_prob']:.2%} "
                      f"actual={metrics['actual_win_rate']:.2%} "
                      f"error={metrics['calibration_error']:.2%}")
                print(f"  Brier Score: {metrics['brier_score']:.4f}")

                if needs_retrain:
                    print(f"  Reason: {reason}")

        print("\n" + "="*80)


    def _get_log_path(self, symbol: str, mode: str) -> str:
        """Get path to prediction log CSV."""
        return os.path.join(self.model_dir, f"{symbol}_{mode}_predictions.csv")


    def cleanup_old_logs(self, days_to_keep: int = 30):
        """
        Remove prediction logs older than specified days.

        Args:
            days_to_keep: Keep logs from last N days
        """
        cutoff = datetime.now() - timedelta(days=days_to_keep)

        for filename in os.listdir(self.model_dir):
            if not filename.endswith('_predictions.csv'):
                continue

            filepath = os.path.join(self.model_dir, filename)

            try:
                df = pd.read_csv(filepath, parse_dates=['timestamp'])

                # Keep only recent predictions
                df_recent = df[df['timestamp'] > cutoff]

                if len(df_recent) < len(df):
                    df_recent.to_csv(filepath, index=False)
                    print(f"[CLEANUP] {filename}: {len(df)} → {len(df_recent)} rows")

            except Exception as e:
                print(f"[WARN] Could not cleanup {filename}: {e}")


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

# Global monitor instance
_monitor = None

def get_monitor() -> ModelMonitor:
    """Get global monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = ModelMonitor()
    return _monitor


def log_prediction(symbol: str, mode: str, prob: float, price: float):
    """Quick logging function."""
    get_monitor().log_prediction(symbol, mode, prob, price)


def evaluate_predictions(symbol: str, mode: str, current_price: float):
    """Quick evaluation function."""
    return get_monitor().evaluate_predictions(symbol, mode, current_price)


def check_needs_retraining(symbol: str, mode: str) -> Tuple[bool, str]:
    """Quick retraining check."""
    return get_monitor().needs_retraining(symbol, mode)


if __name__ == "__main__":
    # Demo
    print("\nModel Monitor Demo")
    print("="*60)

    monitor = ModelMonitor()

    # Simulate some predictions
    print("\n1. Logging predictions...")
    monitor.log_prediction("NVDA", "intraday_mr", 0.75, 185.50)
    monitor.log_prediction("NVDA", "intraday_mom", 0.82, 185.50)

    print("\n2. Performance metrics (will be empty - need evaluated predictions):")
    metrics = monitor.get_performance_metrics("NVDA", "intraday_mr", lookback_days=7)
    print(metrics)

    print("\n3. Retraining check:")
    needs_retrain, reason = monitor.needs_retraining("NVDA", "intraday_mr")
    print(f"Needs retraining: {needs_retrain}")
    print(f"Reason: {reason}")
