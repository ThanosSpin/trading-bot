# class_balancing.py
"""
Handles class imbalance for trading models.
Implements SMOTE, undersampling, and XGBoost scale_pos_weight.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from collections import Counter

def calculate_scale_pos_weight(y: pd.Series) -> float:
    """
    Calculate XGBoost scale_pos_weight parameter.
    
    Formula: scale_pos_weight = count(negative) / count(positive)
    
    This gives higher weight to minority class during training.
    """
    counts = y.value_counts()
    neg_count = counts.get(0, 0)
    pos_count = counts.get(1, 0)
    
    if pos_count == 0:
        print("[WARN] No positive samples in target!")
        return 1.0
    
    weight = neg_count / pos_count
    return weight


def analyze_class_balance(y: pd.Series, name: str = "Dataset"):
    """
    Print detailed class distribution analysis.
    """
    counts = y.value_counts().sort_index()
    total = len(y)
    
    print(f"\n{'='*60}")
    print(f"CLASS BALANCE ANALYSIS: {name}")
    print('='*60)
    
    for cls, count in counts.items():
        pct = (count / total) * 100
        print(f"  Class {cls}: {count:6d} samples ({pct:5.1f}%)")
    
    print(f"\nTotal: {total} samples")
    
    # Imbalance ratio
    if len(counts) == 2:
        ratio = max(counts) / min(counts)
        print(f"Imbalance Ratio: {ratio:.2f}:1")
        
        if ratio > 3:
            print("⚠️  Severe imbalance detected (>3:1) - consider resampling")
        elif ratio > 1.5:
            print("ℹ️  Moderate imbalance (>1.5:1) - use scale_pos_weight")
        else:
            print("✅ Classes fairly balanced")
    
    print('='*60)


def apply_smote(
    X: pd.DataFrame, 
    y: pd.Series, 
    sampling_strategy: float = 0.8,
    k_neighbors: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique).
    
    Args:
        X: Features
        y: Binary target (0/1)
        sampling_strategy: Desired ratio of minority/majority (0.8 = 80%)
        k_neighbors: Number of nearest neighbors for SMOTE
        random_state: Random seed
    
    Returns:
        X_resampled, y_resampled
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        print("[ERROR] imbalanced-learn not installed. Run: pip install imbalanced-learn")
        return X, y
    
    print(f"\n[SMOTE] Applying over-sampling...")
    print(f"  Before: {Counter(y)}")
    
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state
    )
    
    X_res, y_res = smote.fit_resample(X, y)
    
    print(f"  After: {Counter(y_res)}")
    print(f"  Generated {len(X_res) - len(X)} synthetic samples")
    
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


def apply_random_undersampling(
    X: pd.DataFrame,
    y: pd.Series,
    sampling_strategy: float = 0.8,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply random undersampling of majority class.
    
    Args:
        X: Features
        y: Binary target
        sampling_strategy: Desired ratio of minority/majority
        random_state: Random seed
    
    Returns:
        X_resampled, y_resampled
    """
    try:
        from imblearn.under_sampling import RandomUnderSampler
    except ImportError:
        print("[ERROR] imbalanced-learn not installed.")
        return X, y
    
    print(f"\n[UNDERSAMPLE] Applying random undersampling...")
    print(f"  Before: {Counter(y)}")
    
    rus = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )
    
    X_res, y_res = rus.fit_resample(X, y)
    
    print(f"  After: {Counter(y_res)}")
    print(f"  Removed {len(X) - len(X_res)} majority samples")
    
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


def apply_smote_tomek(
    X: pd.DataFrame,
    y: pd.Series,
    sampling_strategy: float = 0.8,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE + Tomek Links (combination).
    
    SMOTE over-samples minority, then Tomek Links clean up boundary samples.
    Best of both worlds for noisy data.
    """
    try:
        from imblearn.combine import SMOTETomek
    except ImportError:
        print("[ERROR] imbalanced-learn not installed.")
        return X, y
    
    print(f"\n[SMOTE+TOMEK] Applying combined resampling...")
    print(f"  Before: {Counter(y)}")
    
    smt = SMOTETomek(
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )
    
    X_res, y_res = smt.fit_resample(X, y)
    
    print(f"  After: {Counter(y_res)}")
    
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


def get_class_weights(y: pd.Series) -> dict:
    """
    Calculate class weights for sklearn models (e.g., Logistic Regression).
    
    Returns:
        {0: weight_0, 1: weight_1}
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    return dict(zip(classes, weights))


def choose_balancing_strategy(
    y: pd.Series,
    strategy: str = 'auto'
) -> Tuple[str, Optional[dict]]:
    """
    Automatically choose best balancing strategy based on data.
    
    Args:
        y: Target variable
        strategy: 'auto', 'scale_pos_weight', 'smote', 'undersample', 'none'
    
    Returns:
        (recommended_strategy, params)
    """
    counts = y.value_counts()
    total = len(y)
    ratio = max(counts) / min(counts) if len(counts) >= 2 else 1.0
    
    if strategy == 'auto':
        if ratio < 1.3:
            return 'none', {}
        elif ratio < 2.0:
            return 'scale_pos_weight', {'scale_pos_weight': calculate_scale_pos_weight(y)}
        elif total < 1000:
            # Small dataset + imbalance -> SMOTE
            return 'smote', {'sampling_strategy': 0.8}
        else:
            # Large dataset -> combine SMOTE + undersample
            return 'smote_undersample', {'over_ratio': 0.7, 'under_ratio': 0.9}
    else:
        if strategy == 'scale_pos_weight':
            return strategy, {'scale_pos_weight': calculate_scale_pos_weight(y)}
        elif strategy == 'smote':
            return strategy, {'sampling_strategy': 0.8}
        elif strategy == 'undersample':
            return strategy, {'sampling_strategy': 0.8}
        elif strategy == 'none':
            return strategy, {}
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    # Demo
    print("\n" + "="*80)
    print("CLASS BALANCING DEMO")
    print("="*80)
    
    # Create synthetic imbalanced dataset
    np.random.seed(42)
    n_samples = 1000
    n_minority = 200
    
    # 80% majority (0), 20% minority (1)
    y = pd.Series([0] * (n_samples - n_minority) + [1] * n_minority)
    X = pd.DataFrame(np.random.randn(n_samples, 10), columns=[f'feature_{i}' for i in range(10)])
    
    # Analyze
    analyze_class_balance(y, "Original Data")
    
    # Calculate scale_pos_weight
    spw = calculate_scale_pos_weight(y)
    print(f"\n✅ XGBoost scale_pos_weight = {spw:.3f}")
    print(f"   Use this in XGBClassifier(scale_pos_weight={spw:.3f})")
    
    # Auto strategy recommendation
    strategy, params = choose_balancing_strategy(y, strategy='auto')
    print(f"\n✅ Recommended strategy: {strategy}")
    print(f"   Parameters: {params}")
    
    # Test SMOTE
    X_smote, y_smote = apply_smote(X, y, sampling_strategy=0.8)
    analyze_class_balance(y_smote, "After SMOTE")
    
    print("\n✅ Class balancing demo complete!")
    print("\nUsage in your model training:")
    print("""
    # Option 1: Use scale_pos_weight (recommended for XGBoost)
    from class_balancing import calculate_scale_pos_weight
    spw = calculate_scale_pos_weight(y_train)
    model = XGBClassifier(scale_pos_weight=spw, ...)
    
    # Option 2: Apply SMOTE before training
    from class_balancing import apply_smote
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    model.fit(X_train_balanced, y_train_balanced)
    """)
