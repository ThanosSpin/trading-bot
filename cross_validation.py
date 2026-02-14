# cross_validation.py
"""
Time-series aware cross-validation for trading models.
Prevents look-ahead bias and data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from typing import Generator, Tuple, List
import matplotlib.pyplot as plt

def time_series_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    gap: int = 5,
    test_size: int = None
) -> Generator[Tuple[pd.Index, pd.Index], None, None]:
    """
    Time-series cross-validation with gap to prevent look-ahead.
    
    Args:
        df: DataFrame with time-indexed data
        n_splits: Number of CV folds
        gap: Number of samples to skip between train/test (prevents leakage)
        test_size: Fixed test size (optional, default is adaptive)
    
    Yields:
        (train_index, test_index) tuples
    
    Example:
        Train: [0:100], Gap: [100:105], Test: [105:130]
        Train: [0:130], Gap: [130:135], Test: [135:160]
        ...
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size)
    
    for train_idx, test_idx in tscv.split(df):
        yield df.index[train_idx], df.index[test_idx]


def purged_kfold_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    embargo_pct: float = 0.01
) -> Generator[Tuple[pd.Index, pd.Index], None, None]:
    """
    Purged K-Fold for overlapping samples (advanced).
    
    Useful when you have overlapping prediction targets
    (e.g., predicting 5-bar forward return creates overlap).
    
    Args:
        df: DataFrame with time-indexed data
        n_splits: Number of folds
        embargo_pct: Percentage of data to embargo after each test set
    
    Yields:
        (train_index, test_index) tuples
    """
    n = len(df)
    embargo_size = int(n * embargo_pct)
    fold_size = n // n_splits
    
    for i in range(n_splits):
        # Test set
        test_start = i * fold_size
        test_end = min((i + 1) * fold_size, n)
        test_idx = df.index[test_start:test_end]
        
        # Train set (everything before test, excluding embargo)
        train_end_1 = max(0, test_start - embargo_size)
        train_idx_1 = df.index[:train_end_1] if train_end_1 > 0 else pd.Index([])
        
        # Can also include data after test + embargo (combinatorial purged)
        # For simplicity, we'll just use past data
        train_idx = train_idx_1
        
        if len(train_idx) > 0 and len(test_idx) > 0:
            yield train_idx, test_idx


def walk_forward_cv(
    df: pd.DataFrame,
    train_size: int,
    test_size: int,
    step_size: int = None
) -> Generator[Tuple[pd.Index, pd.Index], None, None]:
    """
    Walk-forward analysis (rolling window).
    
    Args:
        df: DataFrame
        train_size: Number of samples in training window
        test_size: Number of samples in test window
        step_size: Step between folds (default = test_size)
    
    Yields:
        (train_index, test_index) tuples
    
    Example with train=100, test=20, step=20:
        Fold 1: Train[0:100], Test[100:120]
        Fold 2: Train[20:120], Test[120:140]
        Fold 3: Train[40:140], Test[140:160]
    """
    if step_size is None:
        step_size = test_size
    
    n = len(df)
    start = 0
    
    while start + train_size + test_size <= n:
        train_end = start + train_size
        test_end = train_end + test_size
        
        train_idx = df.index[start:train_end]
        test_idx = df.index[train_end:test_end]
        
        yield train_idx, test_idx
        
        start += step_size


def expanding_window_cv(
    df: pd.DataFrame,
    min_train_size: int,
    test_size: int,
    step_size: int = None
) -> Generator[Tuple[pd.Index, pd.Index], None, None]:
    """
    Expanding window (anchored walk-forward).
    Training set grows over time, test window is fixed.
    
    Args:
        df: DataFrame
        min_train_size: Minimum training samples for first fold
        test_size: Number of samples in test window
        step_size: Step between folds (default = test_size)
    
    Yields:
        (train_index, test_index) tuples
    
    Example:
        Fold 1: Train[0:100], Test[100:120]
        Fold 2: Train[0:120], Test[120:140]  <- train grows
        Fold 3: Train[0:140], Test[140:160]
    """
    if step_size is None:
        step_size = test_size
    
    n = len(df)
    train_end = min_train_size
    
    while train_end + test_size <= n:
        test_end = train_end + test_size
        
        train_idx = df.index[:train_end]
        test_idx = df.index[train_end:test_end]
        
        yield train_idx, test_idx
        
        train_end += step_size


def visualize_cv_splits(
    df: pd.DataFrame,
    cv_generator: Generator,
    title: str = "Cross-Validation Splits"
):
    """
    Visualize train/test splits over time.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    splits = list(cv_generator)
    n_splits = len(splits)
    
    for i, (train_idx, test_idx) in enumerate(splits):
        # Plot train indices
        ax.barh(i, len(train_idx), left=train_idx[0], height=0.4, 
                color='blue', alpha=0.6, label='Train' if i == 0 else '')
        
        # Plot test indices
        ax.barh(i, len(test_idx), left=test_idx[0], height=0.4, 
                color='red', alpha=0.6, label='Test' if i == 0 else '')
    
    ax.set_yticks(range(n_splits))
    ax.set_yticklabels([f'Fold {i+1}' for i in range(n_splits)])
    ax.set_xlabel('Sample Index')
    ax.set_title(title)
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


def evaluate_cv_performance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv_generator: Generator,
    metric_func=None
) -> List[float]:
    """
    Evaluate model performance across CV folds.
    
    Args:
        model: Sklearn-compatible model
        X: Features
        y: Target
        cv_generator: CV generator (from functions above)
        metric_func: Callable(y_true, y_pred) -> score
    
    Returns:
        List of scores for each fold
    """
    from sklearn.metrics import accuracy_score
    
    if metric_func is None:
        metric_func = accuracy_score
    
    scores = []
    
    for train_idx, test_idx in cv_generator:
        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        score = metric_func(y_test, y_pred)
        scores.append(score)
    
    return scores


if __name__ == "__main__":
    # Demo
    from predictive_model.data_loader import fetch_historical_data
    
    print("\n" + "="*80)
    print("TIME SERIES CROSS-VALIDATION DEMO")
    print("="*80)
    
    df = fetch_historical_data("NVDA", period="6mo", interval="1d")
    
    if df is not None and not df.empty:
        print(f"\nData shape: {df.shape}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        # Test TimeSeriesSplit
        print("\n--- TimeSeriesSplit (5 folds, gap=5) ---")
        cv = time_series_cv(df, n_splits=5, gap=5)
        for i, (train_idx, test_idx) in enumerate(cv, 1):
            print(f"Fold {i}: Train={len(train_idx)} samples "
                  f"({train_idx[0].date()} to {train_idx[-1].date()}), "
                  f"Test={len(test_idx)} samples "
                  f"({test_idx[0].date()} to {test_idx[-1].date()})")
        
        # Test Walk-Forward
        print("\n--- Walk-Forward (train=60, test=15, step=15) ---")
        cv = walk_forward_cv(df, train_size=60, test_size=15, step_size=15)
        for i, (train_idx, test_idx) in enumerate(cv, 1):
            print(f"Fold {i}: Train={len(train_idx)} samples, Test={len(test_idx)} samples")
        
        # Test Expanding Window
        print("\n--- Expanding Window (min_train=60, test=15) ---")
        cv = expanding_window_cv(df, min_train_size=60, test_size=15, step_size=15)
        for i, (train_idx, test_idx) in enumerate(cv, 1):
            print(f"Fold {i}: Train={len(train_idx)} samples (expanding), Test={len(test_idx)} samples")
        
        print("\n✅ Cross-validation demos complete!")
    else:
        print("❌ Failed to fetch test data")
