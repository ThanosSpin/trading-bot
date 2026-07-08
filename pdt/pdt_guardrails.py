# pdt_guardrails.py
from pdt.pdt_tracker import get_opened_today_qty


def max_sell_allowed(api, symbol: str, requested_qty: float, margin_status: dict, emergency: bool = False) -> float:
    """
    Returns the maximum SELL qty allowed under intraday margin-aware guardrails.
    0.0 means SELL must be blocked entirely.

    emergency=True:
        - Allows same-day exits even for shares opened today.
        - Used when strategy decides "I must flatten this position now" (risk override).
    """
    symU = str(symbol).upper().strip()

    try:
        requested_qty = float(requested_qty)
    except Exception:
        requested_qty = 0.0

    if requested_qty <= 0:
        return 0.0

    # If we don't have margin_status, fail-open but log.
    if not margin_status:
        print(f"[MARGIN GUARD] No margin_status for {symU}, allowing requested SELL {requested_qty:g}.")
        return requested_qty

    equity = float(margin_status.get("equity", 0.0) or 0.0)
    buying_power = float(margin_status.get("buying_power", 0.0) or 0.0)
    trading_blocked = bool(margin_status.get("trading_blocked", False))

    # If broker says trading is blocked, do not try to adjust; just suppress SELL.
    if trading_blocked:
        print(f"[MARGIN GUARD] Account trading_blocked=true → SELL suppressed for {symU}.")
        return 0.0

    # Shares opened today (intraday exposure)
    opened_today = float(get_opened_today_qty(symU) or 0.0)

    # Broker position
    try:
        pos = api.get_position(symU)
        position_qty = float(getattr(pos, "qty", 0) or 0)
    except Exception:
        position_qty = 0.0

    if position_qty <= 0:
        # Nothing to sell
        return 0.0

    # Shares that are "safe" to sell (not opened today)
    safe_sellable = max(0.0, position_qty - opened_today)

    if safe_sellable <= 0:
        if emergency:
            # Allow same-day exit in emergency mode: close up to full position
            allowed = min(requested_qty, position_qty)
            if allowed > 0:
                print(
                    f"[MARGIN EMERGENCY] Allowing same-day SELL for {symU}: "
                    f"requested={requested_qty:g}, allowed={allowed:g} "
                    f"(position={position_qty:g}, opened_today={opened_today:g})"
                )
                return allowed

        print(
            f"[MARGIN GUARD] SELL blocked for {symU}: "
            f"position={position_qty:g}, opened_today={opened_today:g} (no safe non-intraday shares)."
        )
        return 0.0

    if requested_qty > safe_sellable:
        print(
            f"[MARGIN GUARD] SELL capped for {symU}: "
            f"requested={requested_qty:g}, allowed={safe_sellable:g} "
            f"(opened_today={opened_today:g})"
        )
        return safe_sellable

    return requested_qty