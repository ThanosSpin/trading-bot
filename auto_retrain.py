#!/usr/bin/env python
"""
Auto-retrain models when performance degrades (drift detection)

Features:
- Monitors model accuracy and calibration
- Automatically retrains degraded models
- Validates new models before deployment
- Backs up and restores models if validation fails
- Blacklist for models requiring manual review
"""

import os
import shutil
from datetime import datetime
from predictive_model.model_monitor import get_monitor
from retrain_model import train_intraday_models_with_shap, train_daily_model_with_shap
from config.config import TRAIN_SYMBOLS


# ============================================================
# CONFIGURATION
# ============================================================

ACCURACY_THRESHOLD = 0.52      # Retrain if accuracy drops below 52%
CALIBRATION_THRESHOLD = 0.12   # Retrain if calibration error > 12%
MIN_SAMPLES = 20               # Minimum predictions needed for valid assessment

# Models that require manual review (skip auto-retrain)
RETRAIN_BLACKLIST = {
    ("PLTR", "intraday_mom"),  # Known to produce unstable predictions
    # Add more as needed: ("SYMBOL", "mode")
}


# ============================================================
# MODEL VALIDATION
# ============================================================

def validate_retrained_model(sym: str, mode: str, min_accuracy: float = 0.55) -> bool:
    """
    Validate newly trained model before deploying it.
    
    Args:
        sym: Stock symbol
        mode: Model mode (daily, intraday_mr, intraday_mom)
        min_accuracy: Minimum acceptable accuracy
    
    Returns:
        True if model passes validation, False otherwise
    """
    import joblib
    
    model_path = os.path.join("models", f"{sym}_{mode}_xgb.pkl")
    
    if not os.path.exists(model_path):
        print(f"  ❌ Model file not found: {model_path}")
        return False
    
    try:
        artifact = joblib.load(model_path)
        metrics = artifact.get("metrics", {})
        
        # Extract metrics
        test_acc = metrics.get("accuracy", 0.0)
        test_logloss = metrics.get("logloss", 999.0)
        calibration_error = metrics.get("calibration_error")
        brier_score = metrics.get("brier_score")
        
        print(f"  📊 Validation metrics:")
        print(f"     Accuracy: {test_acc:.1%}")
        print(f"     LogLoss: {test_logloss:.3f}")
        if calibration_error:
            print(f"     Calibration Error: {calibration_error:.1%}")
        if brier_score:
            print(f"     Brier Score: {brier_score:.3f}")
        
        # VALIDATION RULES
        if test_acc < min_accuracy:
            print(f"  ❌ REJECT: Accuracy {test_acc:.1%} < {min_accuracy:.1%}")
            return False
        
        if calibration_error and calibration_error > 0.20:
            print(f"  ❌ REJECT: Calibration error {calibration_error:.1%} > 20%")
            return False
        
        if test_logloss > 0.80:
            print(f"  ❌ REJECT: LogLoss {test_logloss:.3f} > 0.80 (too uncertain)")
            return False
        
        # Additional sanity checks
        if test_acc > 0.95:
            print(f"  ⚠️ WARNING: Suspiciously high accuracy {test_acc:.1%} - possible overfitting")
            # Don't reject, but flag for review
        
        print(f"  ✅ PASS: Model meets quality standards")
        return True
        
    except Exception as e:
        print(f"  ❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# BACKUP/RESTORE UTILITIES
# ============================================================

def backup_existing_model(sym: str, mode: str) -> bool:
    """
    Backup current model before retraining.
    
    Args:
        sym: Stock symbol
        mode: Model mode
    
    Returns:
        True if backup successful, False otherwise
    """
    model_path = os.path.join("models", f"{sym}_{mode}_xgb.pkl")
    backup_path = os.path.join("models", f"{sym}_{mode}_xgb.pkl.backup")
    
    if os.path.exists(model_path):
        try:
            shutil.copy2(model_path, backup_path)
            print(f"  💾 Backed up existing model to {backup_path}")
            return True
        except Exception as e:
            print(f"  ⚠️ Backup failed: {e}")
            return False
    else:
        print(f"  ℹ️ No existing model to backup")
        return False


def restore_backup_model(sym: str, mode: str) -> bool:
    """
    Restore backup if new model fails validation.
    
    Args:
        sym: Stock symbol
        mode: Model mode
    
    Returns:
        True if restore successful, False otherwise
    """
    model_path = os.path.join("models", f"{sym}_{mode}_xgb.pkl")
    backup_path = os.path.join("models", f"{sym}_{mode}_xgb.pkl.backup")
    
    if os.path.exists(backup_path):
        try:
            shutil.copy2(backup_path, model_path)
            print(f"  ↩️ Restored backup model")
            return True
        except Exception as e:
            print(f"  ❌ Restore failed: {e}")
            return False
    else:
        print(f"  ⚠️ No backup found to restore")
        return False


# ============================================================
# MAIN RETRAINING LOGIC
# ============================================================

def check_and_retrain():
    """Check each model's performance and retrain if needed"""
    
    print(f"\n{'='*80}")
    print(f"🔍 AUTO-RETRAIN CHECK - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    monitor = get_monitor()
    symbols_to_retrain = set()
    
    for sym in TRAIN_SYMBOLS:
        print(f"\n📊 Checking {sym}...")
        
        # ============================================================
        # Check daily model
        # ============================================================
        try:
            daily_metrics = monitor.get_performance_metrics(sym, "daily", lookback_days=14)
            
            if daily_metrics['sample_size'] >= MIN_SAMPLES:
                acc = daily_metrics['accuracy']
                cal_err = daily_metrics['calibration_error']
                samples = daily_metrics['sample_size']
                
                print(f"  Daily: {samples} samples, acc={acc:.1%}, cal_err={cal_err:.1%}")
                
                if acc < ACCURACY_THRESHOLD or cal_err > CALIBRATION_THRESHOLD:
                    print(f"  ⚠️ Daily model degraded!")
                    
                    # Check blacklist
                    if (sym, "daily") in RETRAIN_BLACKLIST:
                        print(f"  🚫 {sym} daily is blacklisted - skipping auto-retrain")
                    else:
                        symbols_to_retrain.add((sym, 'daily'))
            else:
                print(f"  Daily: Insufficient data ({daily_metrics['sample_size']} samples)")
                
        except Exception as e:
            print(f"  Daily: Error - {e}")
        
        # ============================================================
        # Check intraday models
        # ============================================================
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
                        
                        # Check blacklist
                        if (sym, mode) in RETRAIN_BLACKLIST:
                            print(f"  🚫 {sym} {mode} is blacklisted - skipping auto-retrain")
                        else:
                            # Add to retrain set (both MR and MOM will be retrained together)
                            symbols_to_retrain.add((sym, 'intraday'))
                            break  # No need to check the other intraday model
                else:
                    print(f"  {mode}: Insufficient data ({metrics['sample_size']} samples)")
                    
            except Exception as e:
                print(f"  {mode}: Error - {e}")
    
    # ============================================================
    # Execute retraining with validation
    # ============================================================
    print(f"\n{'='*80}")
    print(f"🔄 RETRAINING SUMMARY")
    print(f"{'='*80}\n")
    
    if symbols_to_retrain:
        symbols_daily = {s for s, m in symbols_to_retrain if m == 'daily'}
        symbols_intraday = {s for s, m in symbols_to_retrain if m == 'intraday'}
        
        # Retrain daily models
        for sym in symbols_daily:
            print(f"\n{'─'*60}")
            print(f"🔄 Retraining {sym} daily model...")
            print(f"{'─'*60}")
            
            # Backup existing model
            backup_existing_model(sym, "daily")
            
            try:
                train_daily_model_with_shap(sym)
                
                # ✅ VALIDATE before deploying
                if validate_retrained_model(sym, "daily", min_accuracy=0.55):
                    print(f"✅ {sym} daily retrained and validated")
                else:
                    print(f"❌ {sym} daily FAILED VALIDATION - restoring backup")
                    restore_backup_model(sym, "daily")
                    print(f"⚠️ Manual review required for {sym} daily model!")
                    
            except Exception as e:
                print(f"❌ {sym} daily training failed: {e}")
                import traceback
                traceback.print_exc()
                restore_backup_model(sym, "daily")
        
        # Retrain intraday models
        for sym in symbols_intraday:
            print(f"\n{'─'*60}")
            print(f"🔄 Retraining {sym} intraday models (MR + MOM)...")
            print(f"{'─'*60}")
            
            # Backup both MR and MOM models
            backup_existing_model(sym, "intraday_mr")
            backup_existing_model(sym, "intraday_mom")
            
            try:
                train_intraday_models_with_shap(sym)
                
                # ✅ VALIDATE both models
                print(f"\n  Validating intraday_mr...")
                mr_valid = validate_retrained_model(sym, "intraday_mr", min_accuracy=0.55)
                
                print(f"\n  Validating intraday_mom...")
                mom_valid = validate_retrained_model(sym, "intraday_mom", min_accuracy=0.55)
                
                if mr_valid and mom_valid:
                    print(f"\n✅ {sym} intraday models retrained and validated")
                else:
                    if not mr_valid:
                        print(f"\n❌ {sym} intraday_mr FAILED - restoring backup")
                        restore_backup_model(sym, "intraday_mr")
                    if not mom_valid:
                        print(f"❌ {sym} intraday_mom FAILED - restoring backup")
                        restore_backup_model(sym, "intraday_mom")
                    print(f"⚠️ Manual review required for {sym} intraday models!")
                    
            except Exception as e:
                print(f"❌ {sym} intraday training failed: {e}")
                import traceback
                traceback.print_exc()
                restore_backup_model(sym, "intraday_mr")
                restore_backup_model(sym, "intraday_mom")
        
        print(f"\n{'='*80}")
        print(f"✅ Auto-retrain complete!")
        print(f"{'='*80}\n")
    else:
        print("✅ All models performing well - no retraining needed\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        check_and_retrain()
    except Exception as e:
        print(f"\n❌ Auto-retrain failed: {e}")
        import traceback
        traceback.print_exc()
