"""Environment-aware entry allowlist and decision guardrails."""

from __future__ import annotations

from typing import Dict, Optional, Set

from config import (
    BOT_ENV,
    LIVE_ENTRY_SYMBOLS,
    PAPER_ENTRY_SYMBOLS,
    PAPER_TRADE_SYMBOLS,
)


def configured_entry_symbols(environment: Optional[str] = None) -> Set[str]:
    env = str(environment or BOT_ENV).strip().lower()
    configured = PAPER_ENTRY_SYMBOLS if env == "paper" else LIVE_ENTRY_SYMBOLS
    return {str(symbol).strip().upper() for symbol in configured}


def is_new_entry_allowed(symbol: str, environment: Optional[str] = None) -> bool:
    symbol = str(symbol).strip().upper()
    if symbol in {str(item).strip().upper() for item in PAPER_TRADE_SYMBOLS}:
        return True
    return symbol in configured_entry_symbols(environment)


def _is_funding_sell(decision: dict) -> bool:
    if decision.get("funding_sell") or decision.get("rotation_sell"):
        return True
    explanation = str(decision.get("explain", "")).lower()
    return any(
        phrase in explanation
        for phrase in (
            "sold to fund",
            "sell (rotate",
            "rotated into",
            "rotate into",
            "rotation",
        )
    )


def _funding_sell_targets(decision: dict, blocked_symbols: Set[str]) -> Set[str]:
    """Return blocked destination symbols explicitly named by a funding SELL."""
    explanation = str(decision.get("explain", "")).upper()
    return {symbol for symbol in blocked_symbols if symbol in explanation}


def apply_entry_symbol_guard(
    decisions: Dict[str, dict],
    environment: Optional[str] = None,
) -> Dict[str, dict]:
    """Suppress disallowed BUYs while preserving position-closing SELLs."""
    env = str(environment or BOT_ENV).strip().lower()
    if env == "paper":
        return decisions

    guarded = {symbol: dict(decision) for symbol, decision in decisions.items()}
    blocked_buys = []

    for symbol, decision in guarded.items():
        if str(decision.get("action", "")).lower() != "buy":
            continue
        if is_new_entry_allowed(symbol, env):
            continue

        blocked_buys.append(str(symbol).upper())
        prior_reason = str(decision.get("explain", "")).strip()
        decision.update(
            {
                "action": "hold",
                "qty": 0,
                "entry_blocked": True,
                "blocked_original_action": "buy",
                "explain": (
                    f"{str(symbol).upper()}: LIVE BUY blocked by entry allowlist. "
                    f"Signal retained for monitoring. {prior_reason}"
                ).strip(),
            }
        )

    allowed_buy_remains = any(
        str(decision.get("action", "")).lower() == "buy"
        and int(float(decision.get("qty", 0) or 0)) > 0
        for decision in guarded.values()
    )

    blocked_set = set(blocked_buys)
    if blocked_buys:
        for symbol, decision in guarded.items():
            if str(decision.get("action", "")).lower() != "sell":
                continue
            if not _is_funding_sell(decision):
                continue
            named_blocked_targets = _funding_sell_targets(decision, blocked_set)
            # Cancel a sell that explicitly funds a blocked destination. A
            # generic core-funding sell may remain only if an allowed BUY is
            # still scheduled in the same decision batch.
            if not named_blocked_targets and allowed_buy_remains:
                continue
            prior_reason = str(decision.get("explain", "")).strip()
            decision.update(
                {
                    "action": "hold",
                    "qty": 0,
                    "funding_sell_blocked": True,
                    "blocked_original_action": "sell",
                    "explain": (
                        f"{str(symbol).upper()}: funding SELL canceled because "
                        f"all destination BUYs were blocked ({', '.join(blocked_buys)}). "
                        f"{prior_reason}"
                    ).strip(),
                }
            )

    return guarded
