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
    parser.add_argument(
        '--auto-retrain',
        action='store_true',
        help='Automatically retrain models that need it'
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
    performance_summary = []

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
            
            # Track performance
            performance_summary.append({
                'symbol': sym,
                'mode': mode,
                'accuracy': metrics['accuracy'],
                'calibration_error': metrics['calibration_error'],
                'needs_retrain': needs_retrain
            })

    print("\n" + "="*80)

    if retrain_needed:
        print(f"⚠️  {len(retrain_needed)} MODELS NEED RETRAINING:")
        print()
        
        # ✅ UPDATED: Show unified script commands
        for sym, mode in retrain_needed:
            print(f"  • {sym:6} - {mode:12}  →  python apply_week1_optimizations.py --symbol {sym} --mode {mode} --force")
        
        print()
        print("💡 Retrain Options:")
        print("   1. Individual: Run commands above")
        print("   2. Batch:      python apply_week1_optimizations.py --force")
        print("   3. Auto:       python monitor_dashboard.py --auto-retrain")
        
        # ✅ NEW: Auto-retrain feature
        if args.auto_retrain:
            print("\n🔄 AUTO-RETRAIN MODE ENABLED")
            print("="*80)
            
            import subprocess
            
            success_count = 0
            fail_count = 0
            
            for sym, mode in retrain_needed:
                print(f"\n🔨 Retraining {sym}/{mode}...")
                try:
                    result = subprocess.run(
                        ['python', 'apply_week1_optimizations.py', 
                         '--symbol', sym, '--mode', mode, '--force'],
                        capture_output=True,
                        text=True,
                        timeout=600  # 10 minute timeout per model
                    )
                    
                    if result.returncode == 0:
                        print(f"   ✅ {sym}/{mode} retrained successfully")
                        success_count += 1
                    else:
                        print(f"   ❌ {sym}/{mode} failed")
                        print(f"      Error: {result.stderr[:200]}")
                        fail_count += 1
                        
                except subprocess.TimeoutExpired:
                    print(f"   ❌ {sym}/{mode} timed out (>10 minutes)")
                    fail_count += 1
                except Exception as e:
                    print(f"   ❌ {sym}/{mode} error: {e}")
                    fail_count += 1
            
            print("\n" + "="*80)
            print(f"AUTO-RETRAIN SUMMARY: {success_count} success, {fail_count} failed")
            print("="*80)
    else:
        print("✅ ALL MODELS PERFORMING WELL")
        
        # ✅ NEW: Show top/worst performers
        if performance_summary:
            print("\n📊 Performance Summary:")
            
            # Sort by accuracy
            sorted_perf = sorted(performance_summary, key=lambda x: x['accuracy'], reverse=True)
            
            print("\n  🥇 Top Performer:")
            top = sorted_perf[0]
            print(f"     {top['symbol']:6} - {top['mode']:12}  Accuracy: {top['accuracy']:.2%}  |  Cal Error: {top['calibration_error']:.2%}")
            
            if len(sorted_perf) > 1:
                print("\n  🔧 Needs Attention:")
                bottom = sorted_perf[-1]
                print(f"     {bottom['symbol']:6} - {bottom['mode']:12}  Accuracy: {bottom['accuracy']:.2%}  |  Cal Error: {bottom['calibration_error']:.2%}")

    print("="*80)

    # ✅ IMPROVED: Cleanup with stats
    if args.cleanup:
        print("\n🧹 Cleaning up old prediction logs...")
        
        try:
            deleted = monitor.cleanup_old_logs(days_to_keep=30)
            
            if hasattr(deleted, '__len__'):
                print(f"   ✅ Removed {len(deleted)} old log files")
            else:
                print(f"   ✅ Cleanup completed")
                
        except Exception as e:
            print(f"   ⚠️  Cleanup error: {e}")


if __name__ == "__main__":
    main()
