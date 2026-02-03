#!/usr/bin/env python
# monitor_dashboard.py
"""
Model performance monitoring dashboard.
Run this to check if any models need retraining.
"""

import argparse
from datetime import datetime
from model_monitor import ModelMonitor
from config import SYMBOL


def main():
    parser = argparse.ArgumentParser(description="Model Performance Dashboard")
    parser.add_argument(
        '--lookback',
        type=int,
        default=7,
        help='Days to analyze (default: 7)'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up old prediction logs (>30 days)'
    )
    args = parser.parse_args()

    monitor = ModelMonitor()

    print("\n" + "="*80)
    print(f"MODEL PERFORMANCE DASHBOARD - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*80)

    # Get symbols from config
    symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]
    modes = ['daily', 'intraday_mr', 'intraday_mom']

    retrain_needed = []

    for sym in symbols:
        print(f"\n📊 {sym}")
        print("-" * 80)

        for mode in modes:
            metrics = monitor.get_performance_metrics(sym, mode, lookback_days=args.lookback)

            if metrics.get('sample_size', 0) < 5:
                print(f"  {mode:15} → Insufficient data (n={metrics.get('sample_size', 0)})")
                continue

            needs_retrain, reason = monitor.needs_retraining(sym, mode)

            status = "⚠️ RETRAIN" if needs_retrain else "✅ OK"

            print(f"  {mode:15} → {status}")
            print(f"    Samples: {metrics['sample_size']}")
            print(f"    Accuracy: {metrics['accuracy']:.2%}")
            print(f"    Predicted: {metrics['avg_predicted_prob']:.2%} | "
                  f"Actual: {metrics['actual_win_rate']:.2%} | "
                  f"Cal Error: {metrics['calibration_error']:.2%}")
            print(f"    Brier: {metrics['brier_score']:.4f} | "
                  f"LogLoss: {metrics['log_loss']:.4f}")

            if needs_retrain:
                print(f"    ⚠️ {reason}")
                retrain_needed.append((sym, mode))

    print("\n" + "="*80)

    if retrain_needed:
        print(f"⚠️ {len(retrain_needed)} MODELS NEED RETRAINING:")
        for sym, mode in retrain_needed:
            print(f"  - {sym}/{mode}")
        print("\nRun: python retrain_model.py")
    else:
        print("✅ ALL MODELS PERFORMING WELL")

    print("="*80)

    # Cleanup if requested
    if args.cleanup:
        print("\n🧹 Cleaning up old prediction logs...")
        monitor.cleanup_old_logs(days_to_keep=30)


if __name__ == "__main__":
    main()
