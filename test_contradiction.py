
#!/usr/bin/env python
"""
Test script for contradiction detection.
Run after implementing Fix #5.
"""

def test_contradiction_detection():
    try:
        from model_xgb import detect_contradiction
    except ImportError:
        print("❌ detect_contradiction() not found in model_xgb.py")
        print("   You need to add it first (Step 1 of Fix #5)")
        return False

    print("="*60)
    print("TESTING CONTRADICTION DETECTION")
    print("="*60)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Strong momentum + disagreement
    print("\nTest 1: Strong momentum + disagreement (SHOULD detect)")
    result = detect_contradiction(
        daily_prob=0.65,
        intraday_prob=0.35,
        momentum=0.02,
        symbol="TEST",
        verbose=False
    )
    if result['contradiction']:
        print("  ✅ PASS: Contradiction detected")
        print(f"     Severity: {result['severity']}")
        tests_passed += 1
    else:
        print("  ❌ FAIL: Should detect contradiction")
        tests_failed += 1

    # Test 2: Extreme disagreement
    print("\nTest 2: Extreme disagreement (SHOULD detect)")
    result = detect_contradiction(
        daily_prob=0.70,
        intraday_prob=0.30,
        momentum=0.005,
        symbol="TEST",
        verbose=False
    )
    if result['contradiction']:
        print("  ✅ PASS: Contradiction detected")
        print(f"     Severity: {result['severity']}")
        tests_passed += 1
    else:
        print("  ❌ FAIL: Should detect contradiction")
        tests_failed += 1

    # Test 3: Neutral daily + bullish intraday
    print("\nTest 3: Neutral daily + bullish intraday (should NOT detect)")
    result = detect_contradiction(
        daily_prob=0.50,
        intraday_prob=0.80,
        momentum=0.002,
        symbol="TEST",
        verbose=False
    )
    if not result['contradiction']:
        print("  ✅ PASS: No contradiction (correct)")
        tests_passed += 1
    else:
        print("  ❌ FAIL: Should NOT detect contradiction")
        tests_failed += 1

    # Test 4: Models agree
    print("\nTest 4: Models agree (should NOT detect)")
    result = detect_contradiction(
        daily_prob=0.65,
        intraday_prob=0.60,
        momentum=0.02,
        symbol="TEST",
        verbose=False
    )
    if not result['contradiction']:
        print("  ✅ PASS: No contradiction (correct)")
        tests_passed += 1
    else:
        print("  ❌ FAIL: Should NOT detect contradiction")
        tests_failed += 1

    print("\n" + "="*60)
    print(f"RESULTS: {tests_passed}/{tests_passed + tests_failed} tests passed")
    if tests_failed == 0:
        print("✅ ALL TESTS PASSED")
        print("="*60)
        return True
    else:
        print(f"❌ {tests_failed} test(s) failed")
        print("="*60)
        return False

if __name__ == "__main__":
    success = test_contradiction_detection()
    exit(0 if success else 1)
