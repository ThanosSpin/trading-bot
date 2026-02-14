#!/usr/bin/env python
"""
Check if trained models have inverted feature importance
(handles CalibratedClassifierCV wrapper)
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
        
        # ✅ Handle CalibratedClassifierCV wrapper
        if hasattr(model, 'calibrated_classifiers_'):
            print(f"📦 Model is calibrated (CalibratedClassifierCV)")
            # Get base estimator from first calibrated classifier
            base_model = model.calibrated_classifiers_[0].estimator
            print(f"   Base model: {type(base_model).__name__}")
        else:
            base_model = model
        
        # Get feature importances from base model
        if not hasattr(base_model, 'feature_importances_'):
            print(f"⚠️ Base model has no feature_importances_")
            base_model = base_model  # Keep for predictions anyway
            feature_names = artifact.get('feature_names', [])
        else:
            importances = base_model.feature_importances_
            feature_names = artifact.get('feature_names', [])
            
            # Find momentum/return features
            momentum_features = [
                (f, importances[feature_names.index(f)]) 
                for f in feature_names 
                if any(x in f.lower() for x in ['ret', 'mom', 'return', 'pct'])
            ]
            
            if momentum_features:
                print(f"\n📊 Top momentum features:")
                momentum_features.sort(key=lambda x: x[1], reverse=True)
                for feat, imp in momentum_features[:5]:
                    print(f"   {feat:<30s} importance: {imp:.4f}")
        
        # ✅ CRITICAL TEST: Make synthetic predictions
        print(f"\n🧪 Synthetic prediction test:")
        print(f"   (Creating fake data with obvious momentum patterns)")
        
        feature_names = artifact.get('feature_names', [])
        n_features = len(feature_names)
        
        # Find momentum-related features
        mom_candidates = [f for f in feature_names if 'mom' in f.lower() or 'ret_1' in f.lower()]
        
        if not mom_candidates:
            print(f"⚠️ No momentum features found - using ret_1 or Close")
            mom_feature = 'ret_1' if 'ret_1' in feature_names else None
        else:
            mom_feature = mom_candidates[0]
            print(f"   Using feature: {mom_feature}")
        
        # Create dummy feature vectors (median values from training)
        X_base = np.zeros((1, n_features))
        
        # Test 1: Strong POSITIVE momentum
        X_pos = X_base.copy()
        if mom_feature and mom_feature in feature_names:
            idx = feature_names.index(mom_feature)
            X_pos[0, idx] = 0.02  # +2% momentum
        
        prob_pos = model.predict_proba(X_pos)[0][1]  # Use wrapped model
        
        # Test 2: Strong NEGATIVE momentum
        X_neg = X_base.copy()
        if mom_feature and mom_feature in feature_names:
            idx = feature_names.index(mom_feature)
            X_neg[0, idx] = -0.02  # -2% momentum
        
        prob_neg = model.predict_proba(X_neg)[0][1]
        
        # Test 3: Neutral (zero momentum)
        prob_neutral = model.predict_proba(X_base)[0][1]
        
        print(f"\n   Results:")
        print(f"   {'Scenario':<25} {'Predicted Prob':<15} {'Interpretation'}")
        print(f"   {'-'*55}")
        print(f"   {'Positive momentum (+2%)':<25} {prob_pos:.3f}           ", end='')
        if prob_pos > 0.55:
            print(f"Bullish ✅")
        elif prob_pos > 0.45:
            print(f"Neutral ⚠️")
        else:
            print(f"Bearish ❌")
        
        print(f"   {'Neutral momentum (0%)':<25} {prob_neutral:.3f}           ", end='')
        if prob_neutral > 0.55:
            print(f"Bullish")
        elif prob_neutral > 0.45:
            print(f"Neutral")
        else:
            print(f"Bearish")
            
        print(f"   {'Negative momentum (-2%)':<25} {prob_neg:.3f}           ", end='')
        if prob_neg < 0.45:
            print(f"Bearish ✅")
        elif prob_neg < 0.55:
            print(f"Neutral ⚠️")
        else:
            print(f"Bullish ❌")
        
        # ✅ SANITY CHECK
        print(f"\n   📈 Sanity Check:")
        if prob_pos > prob_neutral > prob_neg:
            print(f"   ✅ CORRECT: Positive mom → Higher prob → Negative mom")
            print(f"      Model learned momentum = continuation")
        elif prob_neg > prob_neutral > prob_pos:
            print(f"   ❌ INVERTED: Negative mom → Higher prob → Positive mom")
            print(f"      Model learned momentum = REVERSAL (mean reversion)")
            print(f"      This is WRONG for a 'momentum' model!")
        elif abs(prob_pos - prob_neg) < 0.05:
            print(f"   ⚠️ NO SIGNAL: Model ignores momentum (diff < 5%)")
            print(f"      Model may be using other features only")
        else:
            print(f"   ⚠️ INCONSISTENT: Non-monotonic relationship")
        
        # Show training metrics
        metrics = artifact.get('metrics', {})
        if metrics:
            print(f"\n📊 Training Metrics:")
            print(f"   Accuracy:         {metrics.get('accuracy', 0):.1%}")
            print(f"   ROC-AUC:          {metrics.get('roc_auc', 0):.3f}")
            print(f"   Log Loss:         {metrics.get('logloss', 99):.3f}")
            print(f"   Calibration Err:  {metrics.get('calibration_error', 0):.1%}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()


# Check all your models
print("\n" + "="*60)
print("🔍 MODEL SANITY CHECK - MOMENTUM MODELS")
print("="*60)

for symbol in ['NVDA', 'AAPL', 'ABBV', 'PLTR']:
    try:
        check_model_sanity(symbol, 'intraday_mom')
    except Exception as e:
        print(f"\n❌ Failed to check {symbol}: {e}\n")

print("\n" + "="*60)
print("🔍 MODEL SANITY CHECK - MEAN-REVERSION MODELS")
print("="*60)

for symbol in ['NVDA', 'AAPL', 'ABBV', 'PLTR']:
    try:
        check_model_sanity(symbol, 'intraday_mr')
    except Exception as e:
        print(f"\n❌ Failed to check {symbol}: {e}\n")
