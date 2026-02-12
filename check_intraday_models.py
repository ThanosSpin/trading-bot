#!/usr/bin/env python
"""
Comprehensive Intraday Model Diagnostics

Tests:
1. Time-of-day performance (does accuracy vary by market hour?)
2. Regime switching balance (are both mom/mr models used?)
3. Calibration curves (do probabilities match reality?)
4. Recent prediction analysis (what's happening now?)

Usage:
    python diagnose_intraday_models.py
    python diagnose_intraday_models.py --symbols NVDA AAPL PLTR
    python diagnose_intraday_models.py --days 7  # Last 7 days only
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

# ============================================================
# CONFIGURATION
# ============================================================

LOGS_DIR = "logs"
OUTPUT_DIR = "diagnostics"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# TEST 1: TIME-OF-DAY PERFORMANCE
# ============================================================

def analyze_time_of_day_performance(symbol: str, model_type: str, lookback_days: int = 30):
    """
    Check if model performance varies by market hour.
    Good models should be consistent across all hours.
    """
    print(f"\n{'─'*70}")
    print(f"⏰ TIME-OF-DAY ANALYSIS: {symbol} {model_type}")
    print(f"{'─'*70}")
    
    log_path = os.path.join(LOGS_DIR, f"predictions_{symbol}_{model_type}.csv")
    
    if not os.path.exists(log_path):
        print(f"❌ Log file not found: {log_path}")
        return None
    
    try:
        df = pd.read_csv(log_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Filter recent data
        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=lookback_days)
        df = df[df['timestamp'] > cutoff]
        
        if len(df) == 0:
            print(f"⚠️ No data in last {lookback_days} days")
            return None
        
        # Convert to ET and extract hour
        df['et_time'] = df['timestamp'].dt.tz_convert('America/New_York')
        df['hour'] = df['et_time'].dt.hour
        df['minute'] = df['et_time'].dt.minute
        
        # Filter market hours (9:30 AM - 4:00 PM ET)
        df = df[
            ((df['hour'] == 9) & (df['minute'] >= 30)) |
            ((df['hour'] >= 10) & (df['hour'] < 16))
        ].copy()
        
        print(f"📊 Total predictions: {len(df)}")
        print(f"   Date range: {df['et_time'].min()} to {df['et_time'].max()}")
        
        # Check if we have actual outcomes
        if 'actual_outcome' not in df.columns or df['actual_outcome'].isna().all():
            print(f"⚠️ No actual_outcome data - showing prediction distribution only")
            
            # Show prediction distribution by hour
            hourly_stats = df.groupby('hour')['predicted_prob'].agg(['mean', 'std', 'count'])
            print(f"\n   Predictions by Hour (ET):")
            print(f"   {'Hour':<10} {'Period':<8} {'Mean':<8} {'Std':<8} {'Count':<8}")
            print(f"   {'-'*50}")
            
            for hour in range(9, 17):
                if hour in hourly_stats.index:
                    stats = hourly_stats.loc[hour]
                    period = "OPEN" if hour < 11 else "MIDDAY" if hour < 15 else "CLOSE"
                    print(f"   {hour:2d}:00-{hour+1:2d}:00  {period:<8} {stats['mean']:.3f}    {stats['std']:.3f}    {int(stats['count']):<8}")
            
            return df
        
        # Calculate accuracy by hour
        df_valid = df.dropna(subset=['actual_outcome', 'predicted_prob'])
        
        if len(df_valid) < 20:
            print(f"⚠️ Insufficient labeled data ({len(df_valid)} samples)")
            return df
        
        # Binarize predictions (>0.5 = predict up)
        df_valid['pred_binary'] = (df_valid['predicted_prob'] > 0.5).astype(int)
        df_valid['correct'] = (df_valid['pred_binary'] == df_valid['actual_outcome']).astype(int)
        
        # Group by hour
        hourly_metrics = df_valid.groupby('hour').agg({
            'correct': 'mean',
            'predicted_prob': ['mean', 'std', 'count']
        }).round(3)
        
        print(f"\n   ✅ Accuracy by Hour (ET):")
        print(f"   {'Hour':<10} {'Period':<8} {'Accuracy':<10} {'Avg Prob':<10} {'Count':<8}")
        print(f"   {'-'*60}")
        
        hourly_acc = []
        for hour in range(9, 17):
            if hour in hourly_metrics.index:
                acc = hourly_metrics.loc[hour, ('correct', 'mean')]
                avg_prob = hourly_metrics.loc[hour, ('predicted_prob', 'mean')]
                count = int(hourly_metrics.loc[hour, ('predicted_prob', 'count')])
                period = "OPEN" if hour < 11 else "MIDDAY" if hour < 15 else "CLOSE"
                
                print(f"   {hour:2d}:00-{hour+1:2d}:00  {period:<8} {acc:.1%}        {avg_prob:.3f}      {count:<8}")
                hourly_acc.append(acc)
        
        # Check variance across hours
        if len(hourly_acc) > 0:
            acc_std = np.std(hourly_acc)
            acc_mean = np.mean(hourly_acc)
            
            print(f"\n   Overall: {acc_mean:.1%} ± {acc_std:.1%}")
            
            if acc_std > 0.12:
                print(f"   ⚠️ HIGH VARIANCE ({acc_std:.1%}) - Model unstable across hours")
                print(f"      Consider retraining with more diverse time-of-day data")
            elif acc_std > 0.08:
                print(f"   ⚠️ MODERATE VARIANCE ({acc_std:.1%}) - Some inconsistency")
            else:
                print(f"   ✅ LOW VARIANCE ({acc_std:.1%}) - Consistent performance")
        
        return df_valid
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# TEST 2: REGIME SWITCHING BALANCE
# ============================================================

def analyze_regime_balance(symbol: str, lookback_days: int = 30):
    """
    Check if momentum and mean-reversion models are both used appropriately.
    Healthy balance: 0.5 - 2.0 ratio (both regimes happen regularly)
    """
    print(f"\n{'─'*70}")
    print(f"🔄 REGIME BALANCE ANALYSIS: {symbol}")
    print(f"{'─'*70}")
    
    counts = {}
    recent_data = {}
    
    for model_type in ['intraday_mom', 'intraday_mr']:
        log_path = os.path.join(LOGS_DIR, f"predictions_{symbol}_{model_type}.csv")
        
        if not os.path.exists(log_path):
            print(f"⚠️ {model_type} log not found")
            counts[model_type] = 0
            continue
        
        try:
            df = pd.read_csv(log_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
            # Filter recent data
            cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=lookback_days)
            df_recent = df[df['timestamp'] > cutoff]
            
            counts[model_type] = len(df_recent)
            recent_data[model_type] = df_recent
            
        except Exception as e:
            print(f"⚠️ Error reading {model_type}: {e}")
            counts[model_type] = 0
    
    mom_count = counts.get('intraday_mom', 0)
    mr_count = counts.get('intraday_mr', 0)
    total = mom_count + mr_count
    
    if total == 0:
        print(f"❌ No predictions found for {symbol}")
        return None
    
    print(f"📊 Predictions in last {lookback_days} days:")
    print(f"   Momentum model:      {mom_count:4d} ({100*mom_count/total:.1f}%)")
    print(f"   Mean-reversion model: {mr_count:4d} ({100*mr_count/total:.1f}%)")
    print(f"   Total:                {total:4d}")
    
    # Calculate ratio
    if mr_count > 0:
        ratio = mom_count / mr_count
        print(f"\n   Ratio (mom/mr): {ratio:.2f}")
        
        if ratio < 0.3:
            print(f"   ⚠️ IMBALANCED: Momentum regime rarely triggered")
            print(f"      → Check if mom_trig threshold is too high")
            print(f"      → Or symbol is genuinely mean-reverting")
        elif ratio > 3.0:
            print(f"   ⚠️ IMBALANCED: Mean-reversion regime rarely triggered")
            print(f"      → Check if vol_trig threshold is too high")
            print(f"      → Or symbol is genuinely trending")
        else:
            print(f"   ✅ BALANCED: Both regimes active")
    else:
        print(f"   ⚠️ Mean-reversion model never used!")
    
    # Show recent regime switches
    if 'intraday_mom' in recent_data and 'intraday_mr' in recent_data:
        df_mom = recent_data['intraday_mom']
        df_mr = recent_data['intraday_mr']
        
        if not df_mom.empty and not df_mr.empty:
            # Get last 10 predictions total
            df_mom['regime'] = 'mom'
            df_mr['regime'] = 'mr'
            df_combined = pd.concat([df_mom, df_mr]).sort_values('timestamp').tail(10)
            
            print(f"\n   Recent regime sequence (last 10):")
            for _, row in df_combined.iterrows():
                ts = pd.to_datetime(row['timestamp']).strftime('%m-%d %H:%M')
                regime = row['regime'].upper()
                prob = row.get('predicted_prob', 0.0)
                print(f"      {ts}  [{regime:3s}]  prob={prob:.3f}")
    
    return counts


# ============================================================
# TEST 3: CALIBRATION CURVE
# ============================================================

def plot_calibration_curve(symbol: str, model_type: str, lookback_days: int = 30):
    """
    Check if predicted probabilities match reality.
    A well-calibrated model has predictions that align with actual frequencies.
    """
    print(f"\n{'─'*70}")
    print(f"📈 CALIBRATION ANALYSIS: {symbol} {model_type}")
    print(f"{'─'*70}")
    
    log_path = os.path.join(LOGS_DIR, f"predictions_{symbol}_{model_type}.csv")
    
    if not os.path.exists(log_path):
        print(f"❌ Log file not found: {log_path}")
        return None
    
    try:
        df = pd.read_csv(log_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Filter recent data
        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=lookback_days)
        df = df[df['timestamp'] > cutoff]
        
        # Need actual outcomes
        df = df.dropna(subset=['predicted_prob', 'actual_outcome'])
        
        if len(df) < 30:
            print(f"⚠️ Insufficient labeled data ({len(df)} samples) - need 30+")
            return None
        
        print(f"📊 Analyzing {len(df)} predictions with outcomes")
        
        # Calculate metrics
        y_true = df['actual_outcome'].values
        y_prob = df['predicted_prob'].values
        y_pred = (y_prob > 0.5).astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        brier = brier_score_loss(y_true, y_prob)
        
        try:
            logloss = log_loss(y_true, y_prob)
        except:
            logloss = None
        
        print(f"\n   Performance Metrics:")
        print(f"   {'─'*40}")
        print(f"   Accuracy:      {accuracy:.1%}")
        print(f"   Brier Score:   {brier:.3f} (lower is better)")
        if logloss:
            print(f"   Log Loss:      {logloss:.3f} (lower is better)")
        
        # Compute calibration curve
        n_bins = min(5, len(df) // 10)  # At least 10 samples per bin
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob,
            n_bins=n_bins,
            strategy='quantile'
        )
        
        # Calculate calibration error (ECE)
        calib_error = np.abs(prob_true - prob_pred).mean()
        
        print(f"   Calibration Error: {calib_error:.1%}")
        
        if calib_error > 0.15:
            print(f"   ❌ POOR CALIBRATION - Probabilities are unreliable")
            print(f"      → Retrain with more data")
            print(f"      → Or enable probability calibration (CalibratedClassifierCV)")
        elif calib_error > 0.10:
            print(f"   ⚠️ MODERATE CALIBRATION - Consider recalibrating")
        else:
            print(f"   ✅ WELL-CALIBRATED - Probabilities are trustworthy")
        
        # Plot calibration curve
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
        
        # Actual calibration
        ax.plot(prob_pred, prob_true, 'o-', markersize=8, linewidth=2, 
                label=f'{symbol} {model_type}')
        
        # Formatting
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('Actual Frequency', fontsize=12)
        ax.set_title(f'{symbol} {model_type} - Calibration Curve\n'
                    f'(Accuracy: {accuracy:.1%}, Brier: {brier:.3f}, Cal Error: {calib_error:.1%})',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Save plot
        plot_path = os.path.join(OUTPUT_DIR, f'{symbol}_{model_type}_calibration.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n   💾 Saved plot: {plot_path}")
        
        return {
            'accuracy': accuracy,
            'brier_score': brier,
            'log_loss': logloss,
            'calibration_error': calib_error,
            'n_samples': len(df)
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# TEST 4: RECENT PREDICTIONS SUMMARY
# ============================================================

def analyze_recent_predictions(symbol: str, model_type: str, hours: int = 6):
    """
    Show what the model has been predicting recently.
    Helps spot if model is stuck or behaving strangely.
    """
    print(f"\n{'─'*70}")
    print(f"🕐 RECENT PREDICTIONS: {symbol} {model_type} (last {hours}h)")
    print(f"{'─'*70}")
    
    log_path = os.path.join(LOGS_DIR, f"predictions_{symbol}_{model_type}.csv")
    
    if not os.path.exists(log_path):
        print(f"❌ Log file not found: {log_path}")
        return None
    
    try:
        df = pd.read_csv(log_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        
        # Filter recent hours
        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=hours)
        df_recent = df[df['timestamp'] > cutoff].sort_values('timestamp')
        
        if len(df_recent) == 0:
            print(f"⚠️ No predictions in last {hours} hours")
            return None
        
        print(f"📊 Found {len(df_recent)} predictions")
        
        # Summary statistics
        probs = df_recent['predicted_prob'].dropna()
        
        print(f"\n   Probability Statistics:")
        print(f"   {'─'*40}")
        print(f"   Mean:   {probs.mean():.3f}")
        print(f"   Median: {probs.median():.3f}")
        print(f"   Std:    {probs.std():.3f}")
        print(f"   Min:    {probs.min():.3f}")
        print(f"   Max:    {probs.max():.3f}")
        
        # Check for issues
        if probs.std() < 0.03:
            print(f"\n   ⚠️ WARNING: Very low variance ({probs.std():.3f})")
            print(f"      Model may be stuck or not adapting to conditions")
        
        if probs.mean() > 0.70:
            print(f"\n   ⚠️ WARNING: Strongly bullish-biased (mean {probs.mean():.1%})")
        elif probs.mean() < 0.30:
            print(f"\n   ⚠️ WARNING: Strongly bearish-biased (mean {probs.mean():.1%})")
        
        # Show last 10 predictions
        print(f"\n   Last 10 Predictions:")
        print(f"   {'Timestamp (ET)':<20} {'Prob':<8} {'Price':<10}")
        print(f"   {'-'*40}")
        
        for _, row in df_recent.tail(10).iterrows():
            ts = pd.to_datetime(row['timestamp']).tz_convert('America/New_York').strftime('%m-%d %H:%M:%S')
            prob = row['predicted_prob']
            price = row.get('price', 0.0)
            print(f"   {ts:<20} {prob:.3f}    ${price:.2f}")
        
        return df_recent
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================
# MAIN DIAGNOSTIC RUNNER
# ============================================================

def run_full_diagnostics(symbols, lookback_days=30):
    """
    Run all diagnostic tests for given symbols.
    Generates comprehensive report + visualizations.
    """
    print(f"\n{'='*70}")
    print(f"🔍 INTRADAY MODEL DIAGNOSTICS")
    print(f"{'='*70}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Lookback: {lookback_days} days")
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"{'='*70}")
    
    results = {}
    
    for symbol in symbols:
        print(f"\n\n{'#'*70}")
        print(f"# {symbol}")
        print(f"{'#'*70}")
        
        symbol_results = {}
        
        # Test 2: Regime balance (run first to show overall usage)
        regime_balance = analyze_regime_balance(symbol, lookback_days)
        symbol_results['regime_balance'] = regime_balance
        
        # Run tests for both models
        for model_type in ['intraday_mom', 'intraday_mr']:
            
            # Test 1: Time-of-day performance
            tod_data = analyze_time_of_day_performance(symbol, model_type, lookback_days)
            
            # Test 3: Calibration curve
            calib_metrics = plot_calibration_curve(symbol, model_type, lookback_days)
            
            # Test 4: Recent predictions
            recent_data = analyze_recent_predictions(symbol, model_type, hours=6)
            
            symbol_results[model_type] = {
                'time_of_day': tod_data,
                'calibration': calib_metrics,
                'recent': recent_data
            }
        
        results[symbol] = symbol_results
    
    # Generate summary report
    print(f"\n\n{'='*70}")
    print(f"📋 SUMMARY REPORT")
    print(f"{'='*70}")
    
    for symbol, data in results.items():
        print(f"\n{symbol}:")
        
        for model_type in ['intraday_mom', 'intraday_mr']:
            if model_type in data and data[model_type]['calibration']:
                metrics = data[model_type]['calibration']
                acc = metrics['accuracy']
                cal_err = metrics['calibration_error']
                brier = metrics['brier_score']
                
                status = "✅" if cal_err < 0.10 else "⚠️" if cal_err < 0.15 else "❌"
                
                print(f"  {model_type:15s} {status} Acc: {acc:.1%}  CalErr: {cal_err:.1%}  Brier: {brier:.3f}")
    
    print(f"\n{'='*70}")
    print(f"✅ Diagnostics complete!")
    print(f"📁 Calibration plots saved to: {OUTPUT_DIR}/")
    print(f"{'='*70}\n")
    
    return results


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose intraday model performance")
    parser.add_argument(
        '--symbols', 
        nargs='+', 
        default=['NVDA', 'AAPL', 'ABBV', 'PLTR', 'SPY'],
        help='Symbols to analyze (default: NVDA AAPL ABBV PLTR SPY)'
    )
    parser.add_argument(
        '--days', 
        type=int, 
        default=30,
        help='Days of history to analyze (default: 30)'
    )
    
    args = parser.parse_args()
    
    run_full_diagnostics(args.symbols, args.days)
