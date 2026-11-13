from datetime import datetime, timedelta, timezone

import pandas as pd

from broker import get_active_env, get_trading_api


SYMBOLS = ("NVDA", "AAPL", "ABBV", "PLTR", "SPY")
FEEDS = ("iex", "sip")

api = get_trading_api()
now = datetime.now(timezone.utc)
start = now - timedelta(hours=4)

print(f"BOT_ENV={get_active_env()}  timeframe=15Min")
print(f"Request window: {start.isoformat()} to {now.isoformat()}")
print("Read-only check; no orders will be submitted.\n")

for symbol in SYMBOLS:
    for feed in FEEDS:
        try:
            bars = list(
                api.get_bars(
                    symbol,
                    timeframe="15Min",
                    start=start.isoformat(),
                    end=now.isoformat(),
                    feed=feed,
                    limit=100,
                )
            )

            if not bars:
                print(f"{symbol} {feed}: no bars returned")
                continue

            latest = pd.Timestamp(bars[-1].t)
            if latest.tzinfo is None:
                latest = latest.tz_localize("UTC")
            else:
                latest = latest.tz_convert("UTC")

            age_minutes = (pd.Timestamp.now(tz="UTC") - latest).total_seconds() / 60
            print(
                f"{symbol} {feed}: bars={len(bars)}, "
                f"latest_utc={latest.isoformat()}, age={age_minutes:.1f} min"
            )

        except Exception as exc:
            print(f"{symbol} {feed}: request failed: {type(exc).__name__}: {exc}")