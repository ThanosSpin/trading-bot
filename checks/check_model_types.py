#!/usr/bin/env python3
"""
check_model_types.py

Shows which models are binary vs multi-class and provides
recommendations for retraining.
"""

import os
import joblib
from config.config import MODEL_DIR, TRAIN_SYMBOLS
from tabulate import tabulate
from datetime import datetime
import pandas as pd

def check_model_type(model_path):
    """
    Load model artifact and extract metadata.
    
    Returns:
        dict with model info or None if model doesn't exist
    """
    if not os.path.exists(model_path):
        return None
    
    try:
        artifact = joblib.load(model_path)
        
        # Extract metadata
        info = {
            'exists': True,
            'num_classes': artifact.get('num_classes', 2),
            'target_type': artifact.get('target_type', 'binary'),
            'calibrated': artifact.get('calibrated', False),
            'trained_at': artifact.get('trained_at', 'Unknown'),
            'metrics': artifact.get('metrics', {}),
        }
        
        # Calculate model age
        if info['trained_at'] != 'Unknown':
            try:
                trained_time = datetime.fromisoformat(info['trained_at'])
                age_hours = (datetime.now() - trained_time).total_seconds() / 3600
                info['age_hours'] = age_hours
                info['age_days'] = age_hours / 24
            except:
                info['age_hours'] = None
                info['age_days'] = None
        else:
            info['age_hours'] = None
            info['age_days'] = None
        
        return info
        
    except Exception as e:
        print(f"[ERROR] Failed to load {model_path}: {e}")
        return None


def get_all_model_info():
    """
    Scan MODEL_DIR and check all models.
    
    Returns:
        dict: {symbol: {mode: info}}
    """
    symbols = TRAIN_SYMBOLS if isinstance(TRAIN_SYMBOLS, list) else [TRAIN_SYMBOLS]
    modes = ['daily', 'intraday', 'intraday_mr', 'intraday_mom']
    
    results = {}
    
    for symbol in symbols:
        symbol_upper = symbol.upper()
        results[symbol_upper] = {}
        
        for mode in modes:
            model_path = os.path.join(MODEL_DIR, f"{symbol_upper}_{mode}_xgb.pkl")
            info = check_model_type(model_path)
            results[symbol_upper][mode] = info
    
    return results


def print_model_summary(model_info):
    """
    Print a nice summary table of all models.
    """
    print("\n" + "="*100)
    print("MODEL TYPE SUMMARY")
    print("="*100)
    
    # Build table data
    table_data = []
    
    for symbol, modes in sorted(model_info.items()):
        for mode, info in sorted(modes.items()):
            if info is None:
                row = [
                    symbol,
                    mode,
                    "❌ NOT FOUND",
                    "-",
                    "-",
                    "-",
                    "-"
                ]
            else:
                # Model type with emoji
                if info['target_type'] == 'multiclass':
                    model_type = f"✨ Multi-Class ({info['num_classes']})"
                else:
                    model_type = f"📊 Binary ({info['num_classes']})"
                
                # Calibration status
                calibrated = "✅ Yes" if info['calibrated'] else "❌ No"
                
                # Age
                if info['age_days'] is not None:
                    if info['age_days'] < 1:
                        age = f"{info['age_hours']:.1f}h"
                    else:
                        age = f"{info['age_days']:.1f}d"
                else:
                    age = "Unknown"
                
                # Metrics
                metrics = info.get('metrics', {})
                accuracy = metrics.get('accuracy', metrics.get('accuracy_weighted', 0))
                logloss = metrics.get('logloss', 0)
                
                row = [
                    symbol,
                    mode,
                    model_type,
                    calibrated,
                    age,
                    f"{accuracy:.3f}" if accuracy else "-",
                    f"{logloss:.3f}" if logloss else "-"
                ]
            
            table_data.append(row)
    
    headers = ["Symbol", "Mode", "Type", "Calibrated", "Age", "Accuracy", "LogLoss"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
    print()


def print_statistics(model_info):
    """
    Print overall statistics.
    """
    total_models = 0
    found_models = 0
    binary_models = 0
    multiclass_models = 0
    calibrated_models = 0
    old_models = 0  # >7 days
    
    for symbol, modes in model_info.items():
        for mode, info in modes.items():
            total_models += 1
            
            if info is not None:
                found_models += 1
                
                if info['target_type'] == 'multiclass':
                    multiclass_models += 1
                else:
                    binary_models += 1
                
                if info['calibrated']:
                    calibrated_models += 1
                
                if info.get('age_days') and info['age_days'] > 7:
                    old_models += 1
    
    print("="*100)
    print("STATISTICS")
    print("="*100)
    print(f"Total Expected Models:    {total_models}")
    print(f"Found Models:             {found_models} ({found_models/total_models*100:.0f}%)")
    print(f"Missing Models:           {total_models - found_models}")
    print()
    print(f"📊 Binary Models:         {binary_models}")
    print(f"✨ Multi-Class Models:    {multiclass_models}")
    print(f"✅ Calibrated Models:     {calibrated_models}")
    print(f"⚠️  Old Models (>7d):      {old_models}")
    print("="*100)


def print_recommendations(model_info):
    """
    Print actionable recommendations.
    """
    has_binary = False
    missing_models = []
    old_models = []
    
    for symbol, modes in model_info.items():
        for mode, info in modes.items():
            if info is None:
                missing_models.append(f"{symbol}_{mode}")
            elif info['target_type'] == 'binary':
                has_binary = True
            
            if info and info.get('age_days') and info['age_days'] > 7:
                old_models.append(f"{symbol}_{mode} ({info['age_days']:.1f}d)")
    
    print("\n" + "="*100)
    print("RECOMMENDATIONS")
    print("="*100)
    
    if missing_models:
        print("\n⚠️  MISSING MODELS:")
        for model in missing_models:
            print(f"   - {model}")
        print("\n   Action: Train these models first")
        print("   Command: python apply_week1_optimizations.py")
    
    if has_binary:
        print("\n✨ UPGRADE TO MULTI-CLASS:")
        print("   You have binary models that can be upgraded to multi-class")
        print("   Benefits:")
        print("   - More nuanced predictions (5 classes instead of 2)")
        print("   - Better accuracy (+3-7% expected)")
        print("   - Richer signal details (strong/weak up/down/flat)")
        print("   - Backward compatible (existing code works unchanged)")
        print("\n   Steps:")
        print("   1. Ensure USE_MULTICLASS = True in train_model()")
        print("   2. Run: python apply_week1_optimizations.py")
        print("   3. Test: python test_enhanced_signals.py")
    
    if old_models:
        print("\n⏰ OLD MODELS (Consider Retraining):")
        for model in old_models:
            print(f"   - {model}")
        print("\n   Action: Models >7 days old may be stale")
        print("   Command: python apply_week1_optimizations.py")
    
    if not missing_models and not has_binary and not old_models:
        print("\n✅ ALL MODELS UP TO DATE!")
        print("   - All models are multi-class")
        print("   - All models are recent")
        print("   - No action needed")
    
    print("="*100 + "\n")


def export_to_csv(model_info, filename="model_inventory.csv"):
    """
    Export model info to CSV for record keeping.
    """
    rows = []
    
    for symbol, modes in model_info.items():
        for mode, info in modes.items():
            if info is None:
                row = {
                    'symbol': symbol,
                    'mode': mode,
                    'exists': False,
                    'type': None,
                    'num_classes': None,
                    'calibrated': None,
                    'age_days': None,
                    'accuracy': None,
                    'logloss': None,
                    'trained_at': None,
                }
            else:
                metrics = info.get('metrics', {})
                row = {
                    'symbol': symbol,
                    'mode': mode,
                    'exists': True,
                    'type': info['target_type'],
                    'num_classes': info['num_classes'],
                    'calibrated': info['calibrated'],
                    'age_days': info.get('age_days'),
                    'accuracy': metrics.get('accuracy', metrics.get('accuracy_weighted')),
                    'logloss': metrics.get('logloss'),
                    'trained_at': info['trained_at'],
                }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"📊 Model inventory exported to: {filename}\n")


def check_specific_model(symbol, mode):
    """
    Detailed check of a specific model.
    """
    model_path = os.path.join(MODEL_DIR, f"{symbol.upper()}_{mode}_xgb.pkl")
    
    print(f"\n{'='*100}")
    print(f"DETAILED MODEL CHECK: {symbol.upper()} - {mode}")
    print('='*100)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    try:
        artifact = joblib.load(model_path)
        
        print(f"\n📦 Model Path: {model_path}")
        print(f"📊 File Size: {os.path.getsize(model_path) / 1024:.1f} KB")
        
        print(f"\n🎯 Model Configuration:")
        print(f"   Type: {artifact.get('target_type', 'binary')}")
        print(f"   Classes: {artifact.get('num_classes', 2)}")
        print(f"   Calibrated: {artifact.get('calibrated', False)}")
        print(f"   Trained: {artifact.get('trained_at', 'Unknown')}")
        
        print(f"\n🔧 Features:")
        features = artifact.get('features', [])
        print(f"   Total Features: {len(features)}")
        print(f"   Sample Features: {features[:10]}")
        
        print(f"\n📈 Metrics:")
        metrics = artifact.get('metrics', {})
        for key, value in sorted(metrics.items()):
            if key != 'confusion_matrix':
                try:
                    print(f"   {key}: {value:.4f}")
                except:
                    print(f"   {key}: {value}")
        
        if 'confusion_matrix' in metrics:
            print(f"\n📊 Confusion Matrix:")
            import numpy as np
            cm = np.array(metrics['confusion_matrix'])
            for row in cm:
                print(f"   {row}")
        
        print('='*100 + "\n")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main function - check all models and provide recommendations.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Check model types and status")
    parser.add_argument('--symbol', type=str, help='Check specific symbol')
    parser.add_argument('--mode', type=str, help='Check specific mode (requires --symbol)')
    parser.add_argument('--export', action='store_true', help='Export to CSV')
    
    args = parser.parse_args()
    
    # Specific model check
    if args.symbol and args.mode:
        check_specific_model(args.symbol, args.mode)
        return
    
    # Full scan
    print("\n🔍 Scanning models in:", MODEL_DIR)
    model_info = get_all_model_info()
    
    # Print summary
    print_model_summary(model_info)
    
    # Print statistics
    print_statistics(model_info)
    
    # Print recommendations
    print_recommendations(model_info)
    
    # Export if requested
    if args.export:
        export_to_csv(model_info)
    
    print("💡 TIP: Run with --symbol NVDA --mode daily for detailed model info")
    print("💡 TIP: Run with --export to save inventory as CSV\n")


if __name__ == "__main__":
    main()
