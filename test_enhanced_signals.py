#!/usr/bin/env python3
"""
Test enhanced compute_signals() with multi-class predictions
"""

from model_xgb import compute_signals
import json

def test_signals(symbol="NVDA"):
    print(f"\n{'='*80}")
    print(f"TESTING ENHANCED SIGNALS: {symbol}")
    print(f"{'='*80}\n")
    
    # Compute signals
    results = compute_signals(
        symbol=symbol,
        lookback_minutes=2400,
        intraday_weight=0.65
    )
    
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Basic probabilities (backward compatible)
    print(f"\n📊 Probabilities:")
    print(f"  Daily:    {results.get('daily_prob'):.3f}" if results.get('daily_prob') else "  Daily:    N/A")
    print(f"  Intraday: {results.get('intraday_prob'):.3f}" if results.get('intraday_prob') else "  Intraday: N/A")
    print(f"  Final:    {results.get('final_prob'):.3f}" if results.get('final_prob') else "  Final:    N/A")
    
    # ✨ NEW: Multi-class details
    print(f"\n🎯 Signals:")
    print(f"  Daily:    {results.get('daily_signal', 'N/A')}")
    print(f"  Intraday: {results.get('intraday_signal', 'N/A')}")
    print(f"  Final:    {results.get('final_signal', 'N/A')}")
    
    # Daily prediction details
    daily_pred = results.get('daily_prediction')
    if isinstance(daily_pred, dict):
        print(f"\n📈 Daily Model (Multi-Class):")
        print(f"  Predicted Class: {daily_pred.get('predicted_class_name', 'N/A')}")
        print(f"  Confidence: {daily_pred.get('confidence', 0):.1%}")
        print(f"  Bullish Prob: {daily_pred.get('bullish_prob', 0):.1%}")
        print(f"  Bearish Prob: {daily_pred.get('bearish_prob', 0):.1%}")
        print(f"  Flat Prob: {daily_pred.get('flat_prob', 0):.1%}")
        
        breakdown = daily_pred.get('class_breakdown', {})
        print(f"\n  Class Breakdown:")
        print(f"    Strong Down: {breakdown.get('strong_down', 0):.1%}")
        print(f"    Weak Down:   {breakdown.get('weak_down', 0):.1%}")
        print(f"    Flat:        {breakdown.get('flat', 0):.1%}")
        print(f"    Weak Up:     {breakdown.get('weak_up', 0):.1%}")
        print(f"    Strong Up:   {breakdown.get('strong_up', 0):.1%}")
    
    # Intraday prediction details
    intraday_pred = results.get('intraday_prediction')
    if isinstance(intraday_pred, dict):
        print(f"\n⚡ Intraday Model (Multi-Class):")
        print(f"  Predicted Class: {intraday_pred.get('predicted_class_name', 'N/A')}")
        print(f"  Confidence: {intraday_pred.get('confidence', 0):.1%}")
        print(f"  Bullish Prob: {intraday_pred.get('bullish_prob', 0):.1%}")
        print(f"  Bearish Prob: {intraday_pred.get('bearish_prob', 0):.1%}")
        
        breakdown = intraday_pred.get('class_breakdown', {})
        print(f"\n  Class Breakdown:")
        print(f"    Strong Down: {breakdown.get('strong_down', 0):.1%}")
        print(f"    Weak Down:   {breakdown.get('weak_down', 0):.1%}")
        print(f"    Flat:        {breakdown.get('flat', 0):.1%}")
        print(f"    Weak Up:     {breakdown.get('weak_up', 0):.1%}")
        print(f"    Strong Up:   {breakdown.get('strong_up', 0):.1%}")
    
    # Other diagnostics
    print(f"\n📊 Diagnostics:")
    print(f"  Price: ${results.get('price', 0):.2f}")
    print(f"  Intraday Weight: {results.get('intraday_weight', 0):.2f}")
    print(f"  Intraday Regime: {results.get('intraday_regime', 'N/A')}")
    print(f"  Intraday Vol: {results.get('intraday_vol', 0):.4f}")
    print(f"  Intraday Mom: {results.get('intraday_mom', 0):.2%}")
    print(f"  Model Used: {results.get('intraday_model_used', 'N/A')}")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    test_signals("NVDA")
