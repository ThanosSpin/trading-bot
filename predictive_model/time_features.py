# time_features.py
"""
Time-of-day features for intraday trading models.
Captures market microstructure patterns that repeat daily.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


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


def _market_clock_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return timestamps expressed as naive New York market-clock time.

    Some compatibility and session features retain the caller's original
    index, so they cannot replace it with a New York index.  Convert a
    timezone-aware working copy before extracting dates, hours, or minutes.
    Naive indexes keep their historical interpretation as New York time.
    """
    if index.tz is None:
        return index
    return index.tz_convert(NY_TZ).tz_localize(None)



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


def add_intraday_compatibility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add legacy time/session fields retained for artifact compatibility."""
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    idx = _market_clock_index(df.index)

    clock_minute = idx.hour * 60 + idx.minute
    market_open = 9 * 60 + 30
    market_close = 16 * 60

    df["clock_minute"] = clock_minute.astype(float)
    df["is_regular_session"] = (
        (clock_minute >= market_open)
        & (clock_minute <= market_close)
    ).astype(int)

    minutes_since_open = clock_minute - market_open

    df["is_first_15min"] = (
        (minutes_since_open >= 0)
        & (minutes_since_open < 15)
    ).astype(int)

    df["is_last_15min"] = (
        (minutes_since_open >= 375)
        & (minutes_since_open <= 390)
    ).astype(int)

    return df


def add_intraday_market_features(
    df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add session, VWAP, relative-strength, and benchmark-return features."""
    df = df.copy()

    close = pd.to_numeric(df["Close"], errors="coerce")
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)

    typical_price = (high + low + close) / 3.0

    if isinstance(df.index, pd.DatetimeIndex):
        idx = _market_clock_index(df.index)

        session_key = pd.Series(idx.date, index=df.index)
        cumulative_pv = (typical_price * volume).groupby(session_key).cumsum()
        cumulative_volume = volume.groupby(session_key).cumsum()

        vwap = cumulative_pv / cumulative_volume.replace(0, np.nan)
        df["vwap_distance"] = close / vwap.replace(0, np.nan) - 1.0
        df["above_vwap"] = (close >= vwap).astype(int)

        df["session_volume_pace"] = (
            volume / volume.groupby(session_key).transform("mean").replace(0, np.nan)
        )

        # DatetimeIndex arithmetic returns an Index, and Index does not expose
        # Series.clip().  Keep this indexed so the later volume arithmetic is
        # aligned to the frame rather than relying on positional broadcasting.
        elapsed = pd.Series(
            idx.hour * 60 + idx.minute - (9 * 60 + 30),
            index=df.index,
            dtype=float,
        ).clip(lower=0, upper=390)

        expected_progress = elapsed.div(390.0).clip(lower=1 / 390)
        cumulative_session_volume = volume.groupby(session_key).cumsum()

        session_total_estimate = (
            volume.groupby(session_key).transform("mean") * 26
        )

        expected_volume = session_total_estimate * expected_progress
        df["volume_time_ratio"] = (
            cumulative_session_volume / expected_volume.replace(0, np.nan)
        )

        df["vwap_slope_2"] = df["vwap_distance"].diff(2)
        df["vwap_slope_4"] = df["vwap_distance"].diff(4)
    else:
        df["vwap_distance"] = np.nan
        df["above_vwap"] = 0
        df["session_volume_pace"] = np.nan
        df["volume_time_ratio"] = np.nan
        df["vwap_slope_2"] = np.nan
        df["vwap_slope_4"] = np.nan

    for period in (1, 2, 4):
        df[f"relative_return_{period}"] = np.nan

    df["relative_session_return"] = np.nan
    df["relative_strength_accel"] = np.nan
    df["relative_vwap_strength"] = np.nan
    df["relative_market_available"] = 0
    df["spy_return_1"] = np.nan
    df["spy_return_2"] = np.nan

    if benchmark_df is not None and not benchmark_df.empty:
        benchmark = benchmark_df
        benchmark_close = pd.to_numeric(
            benchmark["Close"], errors="coerce"
        ).dropna()

        if isinstance(df.index, pd.DatetimeIndex):
            output_index = df.index
            target_index = output_index
            bench_index = benchmark_close.index

            if target_index.tz is not None:
                target_index = target_index.tz_convert("UTC").tz_localize(None)
            if bench_index.tz is not None:
                bench_index = bench_index.tz_convert("UTC").tz_localize(None)

            benchmark_close = benchmark_close.copy()
            benchmark_close.index = bench_index
            benchmark_close = (
                benchmark_close[~benchmark_close.index.duplicated(keep="last")]
                .sort_index()
            )

            aligned_benchmark = benchmark_close.reindex(
                target_index,
                method="ffill",
            )
            # Matching is performed on timezone-normalized timestamps, but all
            # derived Series must use the asset frame's original index or
            # pandas will align aware and naive timestamps into all-NaN data.
            aligned_benchmark.index = output_index

            for period in (1, 2, 4):
                asset_return = close.pct_change(period)
                benchmark_return = aligned_benchmark.pct_change(period)

                df[f"relative_return_{period}"] = (
                    asset_return - benchmark_return
                )

            df["spy_return_1"] = aligned_benchmark.pct_change(1)
            df["spy_return_2"] = aligned_benchmark.pct_change(2)

            asset_session = pd.Series(
                output_index.date,
                index=df.index,
            )
            asset_session_return = (
                close / close.groupby(asset_session).transform("first") - 1.0
            )
            benchmark_session_return = (
                aligned_benchmark
                / aligned_benchmark.groupby(asset_session).transform("first")
                - 1.0
            )

            df["relative_session_return"] = (
                asset_session_return - benchmark_session_return
            )

            df["relative_strength_accel"] = (
                df["relative_return_2"].diff()
            )

            df["relative_vwap_strength"] = (
                df["vwap_distance"] - df["relative_session_return"]
            )

            df["relative_market_available"] = (
                aligned_benchmark.notna().astype(int)
            )

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

