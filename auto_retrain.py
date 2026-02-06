#!/usr/bin/env python
"""
Auto-retrain models when performance degrades (drift detection)
"""

from datetime import datetime
from model_monitor import get_monitor
from retrain_model import train_intraday_models_with_shap, train_daily_model_with_shap
from config import TRAIN_SYMBOLS

ACCURACY_THRESHOLD = 0.52
CALIBRATION_THRESHOLD = 0.12
MIN_SAMPLES = 20

def check_and_retrain():
    """Check each model's performance and retrain if needed"""
    
    print(f"\n{'='*80}")
    print(f"🔍 AUTO-RETRAIN CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    monitor = get_monitor()
    symbols_to_retrain = set()
    
    for sym in TRAIN_SYMBOLS:
        print(f"\n📊 Checking {sym}...")
        
        # Check daily model
        try:
            daily_metrics = monitor.get_performance_metrics(sym, "daily", lookback_days=14)
            
            if daily_metrics['sample_size'] >= MIN_SAMPLES:
                acc = daily_metrics['accuracy']
                cal_err = daily_metrics['calibration_error']
                samples = daily_metrics['sample_size']
                
                print(f"  Daily: {samples} samples, acc={acc:.1%}, cal_err={cal_err:.1%}")
                
                if acc < ACCURACY_THRESHOLD or cal_err > CALIBRATION_THRESHOLD:
                    print(f"  ⚠️ Daily model degraded!")
                    symbols_to_retrain.add((sym, 'daily'))
            else:
                print(f"  Daily: Insufficient data ({daily_metrics['sample_size']} samples)")
                
        except Exception as e:
            print(f"  Daily: Error - {e}")
        
        # Check intraday models
        for mode in ["intraday_mr", "intraday_mom"]:
            try:
                metrics = monitor.get_performance_metrics(sym, mode, lookback_days=7)
                
                if metrics['sample_size'] >= MIN_SAMPLES:
                    acc = metrics['accuracy']
                    cal_err = metrics['calibration_error']
                    samples = metrics['sample_size']
                    
                    print(f"  {mode}: {samples} samples, acc={acc:.1%}, cal_err={cal_err:.1%}")
                    
                    if acc < ACCURACY_THRESHOLD or cal_err > CALIBRATION_THRESHOLD:
                        print(f"  ⚠️ {mode} degraded!")
                        symbols_to_retrain.add((sym, 'intraday'))
                        break
                else:
                    print(f"  {mode}: Insufficient data ({metrics['sample_size']} samples)")
                    
            except Exception as e:
                print(f"  {mode}: Error - {e}")
    
    # Execute retraining
    print(f"\n{'='*80}")
    print(f"🔄 RETRAINING SUMMARY")
    print(f"{'='*80}\n")
    
    if symbols_to_retrain:
        symbols_daily = {s for s, m in symbols_to_retrain if m == 'daily'}
        symbols_intraday = {s for s, m in symbols_to_retrain if m == 'intraday'}
        
        for sym in symbols_daily:
            print(f"🔄 Retraining {sym} daily model...")
            try:
                train_daily_model_with_shap(sym)
                print(f"✅ {sym} daily retrained\n")
            except Exception as e:
                print(f"❌ {sym} daily failed: {e}\n")
        
        for sym in symbols_intraday:
            print(f"🔄 Retraining {sym} intraday models...")
            try:
                train_intraday_models_with_shap(sym)
                print(f"✅ {sym} intraday retrained\n")
            except Exception as e:
                print(f"❌ {sym} intraday failed: {e}\n")
        
        print(f"✅ Auto-retrain complete!\n")
    else:
        print("✅ All models performing well\n")


if __name__ == "__main__":
    try:
        check_and_retrain()
    except Exception as e:
        print(f"\n❌ Auto-retrain failed: {e}")
        import traceback
        traceback.print_exc()
