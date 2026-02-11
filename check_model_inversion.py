#!/usr/bin/env python
"""
Check if trained models have inverted feature importance
(e.g., negative momentum predicts up moves)
"""

import joblib
import pandas as pd
import numpy as np

def check_model_sanity(symbol, model_type="intraday_mom"):
    """Check if model learned sensible patterns"""
    
    model_path = f"models/{symbol}_{model_type}_xgb.pkl"
    
    try:
        artifact = joblib.load(model_path)
        model = artifact['model']
        
        print(f"\n{'='*60}")
        print(f"Checking {symbol} {model_type}")
        print(f"{'='*60}")
        
        # Get feature importances
        feature_names = artifact.get('feature_names', [])
        importances = model.feature_importances_
        
        # Find momentum/return features
        momentum_features = [
            f for f in feature_names 
            if any(x in f.lower() for x in ['ret', 'mom', 'return', 'pct'])
        ]
        
        if not momentum_features:
            print("⚠️ No momentum features found in model")
            return
        
        print(f"\n📊 Top momentum features:")
        for feat in momentum_features[:5]:
            if feat in feature_names:
                idx = feature_names.index(feat)
                importance = importances[idx]
                print(f"   {feat:<30s} importance: {importance:.4f}")
        
        # ✅ TEST: Make synthetic predictions
        print(f"\n🧪 Synthetic prediction test:")
        print(f"   (Creating fake data with obvious patterns)")
        
        # Create dummy data (all zeros except one feature)
        n_features = len(feature_names)
        
        # Test 1: Positive momentum
        X_pos = np.zeros((1, n_features))
        if 'ret_1' in feature_names:
            idx_ret1 = feature_names.index('ret_1')
            X_pos[0, idx_ret1] = 0.02  # +2% return
        elif 'mom1h' in feature_names:
            idx_mom = feature_names.index('mom1h')
            X_pos[0, idx_mom] = 0.02
        
        prob_pos = model.predict_proba(X_pos)[0][1]
        
        # Test 2: Negative momentum
        X_neg = np.zeros((1, n_features))
        if 'ret_1' in feature_names:
            X_neg[0, idx_ret1] = -0.02  # -2% return
        elif 'mom1h' in feature_names:
            X_neg[0, idx_mom] = -0.02
        
        prob_neg = model.predict_proba(X_neg)[0][1]
        
        print(f"   Positive momentum (+2%): predicted prob = {prob_pos:.3f}")
        print(f"   Negative momentum (-2%): predicted prob = {prob_neg:.3f}")
        
        # ✅ CHECK SANITY
        if prob_pos > prob_neg:
            print(f"   ✅ CORRECT: Model predicts higher prob for positive momentum")
        else:
            print(f"   ❌ INVERTED: Model predicts LOWER prob for positive momentum!")
            print(f"      This model learned backwards relationships!")
        
        # Check metrics
        metrics = artifact.get('metrics', {})
        print(f"\n📈 Training metrics:")
        print(f"   Accuracy:  {metrics.get('accuracy', 'N/A'):.1%}")
        print(f"   ROC-AUC:   {metrics.get('roc_auc', 'N/A'):.3f}")
        print(f"   Log Loss:  {metrics.get('logloss', 'N/A'):.3f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


# Check all your models
for symbol in ['NVDA', 'AAPL', 'ABBV', 'PLTR']:
    check_model_sanity(symbol, 'intraday_mom')
