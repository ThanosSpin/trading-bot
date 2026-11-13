#!/usr/bin/env python3
"""Read-only diagnostic for the active Alpaca account's latest quotes."""

import os
import re
from datetime import datetime, timezone

from dotenv import load_dotenv
from alpaca_trade_api import REST

load_dotenv()


def _field(obj, *names):
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def main():
    symbols = [
        symbol.strip().upper()
        for symbol in os.getenv("QUOTE_CHECK_SYMBOLS", "AAPL").split(",")
        if symbol.strip()
    ]
    env = os.getenv("BOT_ENV", "live").strip().lower()
    credential_names = {
        "paper": ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_BASE_URL"),
        "live": (
            "ALPACA_MARKET_API_KEY",
            "ALPACA_MARKET_SECRET_KEY",
            "ALPACA_MARKET_BASE_URL",
        ),
        "live2": (
            "ALPACA_LIVE2_MARKET_API_KEY",
            "ALPACA_LIVE2_MARKET_SECRET_KEY",
            "ALPACA_LIVE2_MARKET_BASE_URL",
        ),
    }
    if env not in credential_names:
        raise ValueError("BOT_ENV must be paper, live, or live2")
    key_name, secret_name, url_name = credential_names[env]
    key, secret, base_url = (
        os.getenv(key_name),
        os.getenv(secret_name),
        os.getenv(url_name),
    )
    if not (key and secret and base_url):
        raise RuntimeError(
            f"Missing Alpaca credentials for BOT_ENV={env}; "
            f"check {key_name}, {secret_name}, and {url_name} in the environment/.env"
        )
    api = REST(key, secret, base_url, api_version="v2")
    print(f"Read-only latest-quote check; BOT_ENV={env}; no orders will be submitted.")
    print(f"Checked at {datetime.now(timezone.utc).isoformat()}")

    for symbol in symbols:
        try:
            quote = api.get_latest_quote(symbol)
            bid = _field(quote, "bidprice", "bid_price")
            ask = _field(quote, "askprice", "ask_price")
            timestamp = _field(quote, "timestamp", "t")
            exchange = _field(quote, "exchange", "x")
            if hasattr(timestamp, "isoformat"):
                timestamp = timestamp.isoformat()
            age_text = "unavailable"
            try:
                parsed_timestamp = timestamp
                if isinstance(parsed_timestamp, str):
                    parsed_timestamp = re.sub(
                        r"(\.\d{6})\d+(?=[+-]\d{2}:?\d{2}$|Z$)",
                        r"\1",
                        parsed_timestamp,
                    )
                    parsed_timestamp = datetime.fromisoformat(
                        parsed_timestamp.replace("Z", "+00:00")
                    )
                if parsed_timestamp.tzinfo is None:
                    parsed_timestamp = parsed_timestamp.replace(tzinfo=timezone.utc)
                age_text = f"{(datetime.now(timezone.utc) - parsed_timestamp).total_seconds():.1f}s"
            except (AttributeError, TypeError, ValueError):
                pass
            print(
                f"{symbol}: bid={bid!r} ask={ask!r} "
                f"timestamp={timestamp!r} age={age_text} exchange={exchange!r}"
            )
        except Exception as exc:
            # The exception is limited to quote retrieval; no order methods are called.
            print(f"{symbol}: quote request failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
