# target_engineering.py
"""
Advanced target engineering for trading models.
Creates multi-class and probabilistic targets instead of simple binary.
"""

import pandas as pd
import numpy as np
from typing import Literal

def create_binary_target(df: pd.DataFrame, forward_periods: int = 1) -> pd.DataFrame:
    """
    Simple binary target (your current approach).
    Target = 1 if next return > 0, else 0.
    """
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['Target'] = (df['Return'].shift(-forward_periods) > 0).astype(int)
    return df


def create_multiclass_target(
    df: pd.DataFrame, 
    forward_periods: int = 1,
    thresholds: tuple = (-0.01, -0.002, 0.002, 0.01)
) -> pd.DataFrame:
    """
    Multi-class target for different market regimes.
    
    Classes:
    0: Strong Down (return < -1%)
    1: Weak Down (-1% <= return < -0.2%)
    2: Flat (-0.2% <= return <= 0.2%)
    3: Weak Up (0.2% < return <= 1%)
    4: Strong Up (return > 1%)
    
    Args:
        df: DataFrame with OHLCV data
        forward_periods: Bars ahead to predict
        thresholds: (strong_down, weak_down, weak_up, strong_up)
    
    Returns:
        df with 'Target_multiclass' column (0-4)
    """
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    forward_return = df['Return'].shift(-forward_periods)
    
    strong_down, weak_down, weak_up, strong_up = thresholds
    
    conditions = [
        forward_return < strong_down,           # 0: Strong Down
        (forward_return >= strong_down) & (forward_return < weak_down),  # 1: Weak Down
        (forward_return >= weak_down) & (forward_return <= weak_up),     # 2: Flat
        (forward_return > weak_up) & (forward_return <= strong_up),      # 3: Weak Up
        forward_return > strong_up              # 4: Strong Up
    ]
    
    df['Target_multiclass'] = np.select(conditions, [0, 1, 2, 3, 4], default=2)
    df['Target_multiclass'] = df['Target_multiclass'].astype(int)
    
    return df


def create_soft_labels(
    df: pd.DataFrame, 
    forward_periods: int = 1,
    temperature: float = 0.005
) -> pd.DataFrame:
    """
    Probabilistic 'soft' labels using sigmoid transformation.
    Converts returns to smooth probabilities instead of hard 0/1.
    
    Benefits:
    - Smoother gradients during training
    - Captures magnitude of moves (0.1% vs 2% both get label=1 in binary)
    - Better for regression-like problems
    
    Args:
        df: DataFrame with OHLCV data
        forward_periods: Bars ahead to predict
        temperature: Controls steepness (lower = more sigmoid-like)
    
    Returns:
        df with 'Target_soft' column (0.0 to 1.0)
    """
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    forward_return = df['Return'].shift(-forward_periods)
    
    # Sigmoid transformation: 1 / (1 + exp(-x/temperature))
    df['Target_soft'] = 1 / (1 + np.exp(-forward_return / temperature))
    
    return df


def create_magnitude_aware_target(
    df: pd.DataFrame, 
    forward_periods: int = 1,
    bins: int = 10
) -> pd.DataFrame:
    """
    Creates targets based on return quantiles (magnitude-aware).
    
    Returns:
        df with 'Target_quantile' (0 to bins-1)
    """
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    forward_return = df['Return'].shift(-forward_periods)
    
    df['Target_quantile'] = pd.qcut(
        forward_return, 
        q=bins, 
        labels=False, 
        duplicates='drop'
    )
    
    return df


def create_risk_adjusted_target(
    df: pd.DataFrame, 
    forward_periods: int = 1,
    lookback_vol: int = 20
) -> pd.DataFrame:
    """
    Target normalized by recent volatility (Sharpe-like).
    Better for comparing across different volatility regimes.
    
    Target = forward_return / recent_volatility
    """
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    forward_return = df['Return'].shift(-forward_periods)
    
    # Rolling volatility
    recent_vol = df['Return'].rolling(window=lookback_vol).std()
    
    # Risk-adjusted return
    df['Target_risk_adjusted'] = forward_return / recent_vol
    
    # Convert to binary (positive risk-adjusted return = 1)
    df['Target_risk_adjusted_binary'] = (df['Target_risk_adjusted'] > 0).astype(int)
    
    return df


def create_all_targets(
    df: pd.DataFrame,
    forward_periods: int = 1,
    target_type: Literal['binary', 'multiclass', 'soft', 'magnitude', 'risk_adjusted'] = 'multiclass'
) -> pd.DataFrame:
    """
    Unified interface to create any target type.
    
    Args:
        df: DataFrame with OHLCV
        forward_periods: Prediction horizon
        target_type: Which target to create
    
    Returns:
        df with appropriate Target column
    """
    if target_type == 'binary':
        return create_binary_target(df, forward_periods)
    elif target_type == 'multiclass':
        return create_multiclass_target(df, forward_periods)
    elif target_type == 'soft':
        return create_soft_labels(df, forward_periods)
    elif target_type == 'magnitude':
        return create_magnitude_aware_target(df, forward_periods)
    elif target_type == 'risk_adjusted':
        return create_risk_adjusted_target(df, forward_periods)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")


def print_target_distribution(df: pd.DataFrame, target_col: str = 'Target_multiclass'):
    """
    Print distribution of target classes for debugging.
    """
    if target_col not in df.columns:
        print(f"❌ Column '{target_col}' not found")
        return
    
    print(f"\n{'='*60}")
    print(f"TARGET DISTRIBUTION: {target_col}")
    print('='*60)
    
    counts = df[target_col].value_counts().sort_index()
    total = len(df[target_col].dropna())
    
    for cls, count in counts.items():
        pct = (count / total) * 100
        print(f"  Class {cls}: {count:5d} samples ({pct:5.1f}%)")
    
    print(f"\nTotal: {total} samples")
    print(f"Missing: {df[target_col].isna().sum()} samples")
    print('='*60)


if __name__ == "__main__":
    # Demo/Test
    from predictive_model.data_loader import fetch_historical_data
    
    print("\n" + "="*80)
    print("TARGET ENGINEERING DEMO")
    print("="*80)
    
    df = fetch_historical_data("NVDA", period="3mo", interval="1d")
    
    if df is not None and not df.empty:
        print(f"\nOriginal data shape: {df.shape}")
        
        # Test each target type
        print("\n--- Binary Target ---")
        df_binary = create_binary_target(df.copy())
        print(df_binary[['Close', 'Return', 'Target']].tail())
        print(f"Class distribution: {df_binary['Target'].value_counts().to_dict()}")
        
        print("\n--- Multi-Class Target ---")
        df_multi = create_multiclass_target(df.copy())
        print(df_multi[['Close', 'Return', 'Target_multiclass']].tail())
        print_target_distribution(df_multi, 'Target_multiclass')
        
        print("\n--- Soft Labels ---")
        df_soft = create_soft_labels(df.copy())
        print(df_soft[['Close', 'Return', 'Target_soft']].tail())
        print(f"Soft label stats: mean={df_soft['Target_soft'].mean():.3f}, std={df_soft['Target_soft'].std():.3f}")
        
        print("\n--- Risk-Adjusted Target ---")
        df_risk = create_risk_adjusted_target(df.copy())
        print(df_risk[['Close', 'Return', 'Target_risk_adjusted', 'Target_risk_adjusted_binary']].tail())
        
        print("\n✅ All target types created successfully!")
    else:
        print("❌ Failed to fetch test data")
