# intraday_regime.py
import pandas as pd

def detect_intraday_regime(df_15m: pd.DataFrame) -> str:
    """
    Returns:
      "momentum" or "mean_reversion"

    df_15m: intraday bars at 15m with a Close column.
    """
    if df_15m is None or df_15m.empty or "Close" not in df_15m.columns:
        return "mean_reversion"

    close = df_15m["Close"].dropna()
    if len(close) < 12:
        return "mean_reversion"

    # momentum over last ~2 hours (8 bars)
    mom = (close.iloc[-1] / close.iloc[-9]) - 1.0 if len(close) >= 9 else 0.0

    # realized vol over recent window
    vol = close.pct_change().dropna().tail(48).std()  # last ~12h

    # Tune thresholds per symbol if needed
    if (mom >= 0.010) or (vol >= 0.012):
        return "momentum"

    return "mean_reversion"