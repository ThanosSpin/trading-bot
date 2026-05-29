# time_features.py
"""
Time-of-day features for intraday trading models.
Captures market microstructure patterns that repeat daily.
"""

import pandas as pd
import numpy as np
from typing import Dict

######## Helper  functions ########
NY_TZ = "America/New_York"
MARKET_OPEN_MIN = 9 * 60 + 30
MARKET_CLOSE_MIN = 16 * 60
FULL_SESSION_MIN = MARKET_CLOSE_MIN - MARKET_OPEN_MIN  # 390

def _ensure_ny_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("add_time_features requires a DatetimeIndex")

    out = df.copy()

    if out.index.tz is None:
        raise ValueError(
            "DatetimeIndex must be timezone-aware. "
            "Pass UTC or market-time timestamps explicitly."
        )

    out.index = out.index.tz_convert(NY_TZ)
    return out



def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add comprehensive time-of-day features.

    Market microstructure insights:
    - Opening 30min: High volatility, news-driven, mean reversion dominant
    - Midday (10:30-15:00): Lower volume, trend continuation
    - Lunch (12-1 PM): Lowest liquidity, avoid trading
    - Closing hour: Volume surge, institutional flows, reversals

    Args:
        df: DataFrame with DatetimeIndex (must be timezone-aware or US market hours)

    Returns:
        DataFrame with 15 new time features

    Features created:
        - time_of_day: Decimal hour (9.5 = 9:30 AM)
        - minutes_since_open: Minutes since 9:30 AM (0-390)
        - minutes_to_close: Minutes until 4:00 PM (390-0)
        - is_opening/midday/closing: Session phase indicators
        - time_squared/cubed: Non-linear time effects
        - time_sin/cos: Cyclical encoding
        - is_lunch_hour: Low liquidity period
        - is_first_hour/last_hour: Extreme behavior periods
        - normalized_time: 0-1 scaled time (for neural nets)

    Example:
        >>> df = fetch_historical_data("NVDA", period="5d", interval="15m")
        >>> df = add_time_features(df)
        >>> print(df[['Close', 'time_of_day', 'is_opening', 'minutes_since_open']].head())
    """
    if df.empty:
        return df
    
    df = _ensure_ny_index(df)
    df = df.copy()

    # Extract time components from index
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute

    # === Basic Time Features ===

    df['clock_minute'] = df['hour'] * 60 + df['minute']
    
    # Distance from market open (9:30 AM = 9.5)
    df['minutes_since_open'] = (df['clock_minute'] - MARKET_OPEN_MIN).clip(lower=0)
    df['minutes_to_close'] = (MARKET_CLOSE_MIN - df['clock_minute']).clip(lower=0)

    df['time_of_day'] = df['clock_minute'] / 60.0
    # df['normalized_time'] = (df['minutes_since_open'] / FULL_SESSION_MIN).clip(0, 1)

    # === Session Phase (One-Hot Encoded) ===

    # def get_session_phase(minutes: float) -> str:
    #     """Classify into opening/midday/closing session."""
    #     if minutes <= 30:
    #         return 'opening'
    #     elif minutes >= 330:  # Last 60 minutes (5.5h from open)
    #         return 'closing'
    #     else:
    #         return 'midday'

    # df['session_phase'] = df['minutes_since_open'].apply(get_session_phase)

    df['is_regular_session'] = (
        (df['clock_minute'] >= MARKET_OPEN_MIN) &
        (df['clock_minute'] < MARKET_CLOSE_MIN)
    ).astype(int)

    # One-hot encode (drop session_phase after)
    df['is_opening'] = (
        (df['minutes_since_open'] <= 30) &
        (df['is_regular_session'] == 1)
    ).astype(int)

    df['is_closing'] = (
        (df['minutes_since_open'] >= 330) &
        (df['is_regular_session'] == 1)
    ).astype(int)

    df['is_midday'] = (
        (df['is_regular_session'] == 1) &
        (df['is_opening'] == 0) &
        (df['is_closing'] == 0)
    ).astype(int)
    
    # Lunch hour (12:00-13:00) - low liquidity, avoid trading
    df['is_lunch_hour'] = (
        (df['clock_minute'] >= 12 * 60) &
        (df['clock_minute'] < 13 * 60) &
        (df['is_regular_session'] == 1)
    ).astype(int)

    # === Non-Linear Time Effects ===

    # # Polynomial features (capture acceleration/deceleration)
    # df['time_squared'] = df['minutes_since_open'] ** 2
    # df['time_cubed'] = df['minutes_since_open'] ** 3

    # Cyclical encoding (sine/cosine for full trading day cycle)
    # 390 minutes = full cycle
    session_pos = df['minutes_since_open'].clip(0, FULL_SESSION_MIN) / FULL_SESSION_MIN
    df['time_sin'] = np.where(
        df['is_regular_session'] == 1,
        np.sin(2 * np.pi * session_pos),
        0.0
    )
    df['time_cos'] = np.where(
        df['is_regular_session'] == 1,
        np.cos(2 * np.pi * session_pos),
        0.0
    )


    # === Special Period Indicators ===
    df['is_first_hour'] = (
        (df['minutes_since_open'] <= 60) &
        (df['is_regular_session'] == 1)
    ).astype(int)

    df['is_last_hour'] = (
        (df['minutes_to_close'] <= 60) &
        (df['is_regular_session'] == 1)
    ).astype(int)

    df['is_first_15min'] = (
        (df['minutes_since_open'] <= 15) &
        (df['is_regular_session'] == 1)
    ).astype(int)

    df['is_last_15min'] = (
        (df['minutes_to_close'] <= 15) &
        (df['is_regular_session'] == 1)
    ).astype(int)

    # === Cleanup ===

    # Drop intermediate columns
    df = df.drop(columns=['hour', 'minute', 'session_phase'], errors='ignore')

    return df


def get_time_feature_stats(df: pd.DataFrame) -> Dict[str, any]:
    """
    Analyze time feature patterns in data.
    Useful for understanding market microstructure.

    Args:
        df: DataFrame with time features already added

    Returns:
        Dictionary with statistics by session phase
    """
    if 'is_opening' not in df.columns:
        raise ValueError("Run add_time_features() first")

    stats = {}

    # Returns by session
    work = df.copy()
    work['forward_ret'] = work['Close'].pct_change().shift(-1)


    for phase in ['opening', 'midday', 'closing']:
        mask = work[f'is_{phase}'] == 1
        phase_data = work[mask]

        if len(phase_data) > 0:
            stats[phase] = {
                'count': len(phase_data),
                'avg_return': phase_data['forward_ret'].mean(),
                'volatility': phase_data['forward_ret'].std(),
                'avg_volume': phase_data['Volume'].mean() if 'Volume' in work.columns else None,
                'win_rate': (phase_data['forward_ret'] > 0).mean()
            }

    return stats


def print_time_feature_summary(df: pd.DataFrame):
    """Print summary of time features for debugging."""
    print("\n" + "="*60)
    print("TIME FEATURES SUMMARY")
    print("="*60)

    if 'time_of_day' not in df.columns:
        print("⚠️ Time features not added yet. Run add_time_features() first.")
        return

    time_cols = [
        'time_of_day', 'minutes_since_open', 'minutes_to_close',
        'is_opening', 'is_midday', 'is_closing', 'is_lunch_hour',
        'is_first_hour', 'is_last_hour'
    ]

    available = [col for col in time_cols if col in df.columns]

    print(f"\n📊 Sample (first 5 rows):")
    print(df[available].head())

    print(f"\n📈 Session Distribution:")
    print(f"  Opening: {df['is_opening'].sum()} bars ({df['is_opening'].mean()*100:.1f}%)")
    print(f"  Midday:  {df['is_midday'].sum()} bars ({df['is_midday'].mean()*100:.1f}%)")
    print(f"  Closing: {df['is_closing'].sum()} bars ({df['is_closing'].mean()*100:.1f}%)")

    if 'Close' in df.columns:
        stats = get_time_feature_stats(df)
        print(f"\n💹 Returns by Session:")
        for phase, data in stats.items():
            print(f"  {phase.capitalize():8} → avg_ret={data['avg_return']*100:+.3f}% vol={data['volatility']*100:.3f}% win_rate={data['win_rate']*100:.1f}%")


if __name__ == "__main__":
    # Demo
    print("\nTime Features Demo")
    print("="*60)

    # Create sample data
    ny_tz = "America/New_York"

    dates = pd.date_range(
        start="2026-02-03 09:30:00",
        periods=26,
        freq="15min",
        tz=ny_tz
    )

    df = pd.DataFrame({
        'Close': np.random.randn(26).cumsum() + 100,
        'Volume': np.random.randint(1_000_000, 5_000_000, 26)
    }, index=dates)

    print(f"\nBefore: {df.shape[1]} columns")
    print(df.columns.tolist())

    df = add_time_features(df)

    print(f"\nAfter: {df.shape[1]} columns")
    print(df.columns.tolist())

    print_time_feature_summary(df)

