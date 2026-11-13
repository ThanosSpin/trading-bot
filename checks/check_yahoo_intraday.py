from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

SYMBOLS = ("NVDA", "AAPL", "ABBV", "PLTR", "SPY")
NY = ZoneInfo("America/New_York")

print("Read-only Yahoo data check; no orders will be placed.")
print("Current New York time:", datetime.now(NY).isoformat())

for symbol in SYMBOLS:
    for interval in ("1m", "15m"):
        try:
            df = yf.download(
                symbol,
                period="1d",
                interval=interval,
                prepost=True,
                progress=False,
                auto_adjust=False,
                threads=False,
            )

            if df is None or df.empty:
                print(f"{symbol} {interval}: no data returned")
                continue

            times = pd.DatetimeIndex(pd.to_datetime(df.index))
            if times.tz is not None:
                times = times.tz_convert(NY)
                newest = times[-1].isoformat()
                age_min = (datetime.now(NY) - times[-1].to_pydatetime()).total_seconds() / 60
            else:
                newest = f"{times[-1]} (timezone missing; age not calculated)"
                age_min = None

            spacing = times.to_series().diff().dropna().dt.total_seconds() / 60
            median_spacing = spacing.median() if not spacing.empty else None

            print(
                f"{symbol} {interval}: rows={len(df)}, "
                f"latest={newest}, "
                f"age_min={None if age_min is None else round(age_min, 1)}, "
                f"median_spacing_min="
                f"{None if median_spacing is None else round(float(median_spacing), 1)}"
            )

        except Exception as exc:
            print(f"{symbol} {interval}: {type(exc).__name__}: {exc}")