from typing import Dict, List, Optional
from config.config import (
    BUY_THRESHOLD, SELL_THRESHOLD, STOP_LOSS, RISK_FRACTION,
    SPY_SYMBOL, WEAK_PROB_THRESHOLD, WEAK_RATIO_THRESHOLD, TRAIL_ACTIVATE,
    SPY_ENTRY_THRESHOLD, SPY_EXIT_THRESHOLD, SPY_MUTUAL_EXCLUSIVE, SPY_RISK_FRACTION, TRAIL_STOP,
    PDT_TIERING_ENABLED, PDT_EMERGENCY_STOP, RS_MARGIN, MAX_POSITION_SIZE_PCT,
    MAX_POSITION_SIZE_DOLLARS, DIP_BUY_ENABLED, DIP_BUY_THRESHOLD, DIP_BUY_MIN_PROB,
    PYRAMID_THRESHOLD
)
from portfolio import PortfolioManager
from predictive_model.data_loader import fetch_latest_price, fetch_historical_data
from trader import get_pdt_status
from pdt.pdt_tracker import get_opened_today_qty
from account_cache import account_cache



# ---------------------------------------------------------
# Helper for afterhours_dip
# ---------------------------------------------------------
def detect_afterhours_dip(sym: str, threshold: float = 0.015) -> bool:
    """
    Returns True if stock dropped >= threshold after market close.
    Compares current pre-market price vs yesterday's close.
    """
    try:
        from predictive_model.data_loader import fetch_historical_data, fetch_latest_price

        # Get yesterday's close
        df = fetch_historical_data(sym, period="5d", interval="1d")
        if df is None or len(df) < 1:
            return False

        last_close = float(df["Close"].iloc[-1])

        # Get current price (pre-market or live)
        current = fetch_latest_price(sym)
        if current is None:
            return False

        drop_pct = (current - last_close) / last_close

        print(f"[DIP] {sym} close={last_close:.2f} current={current:.2f} drop={drop_pct:.2%}")

        return drop_pct <= -threshold  # True if dropped >= 1.5%

    except Exception as e:
        print(f"[ERROR] detect_afterhours_dip: {e}")
        return False


# ---------------------------------------------------------
# Helper for position limits
# ---------------------------------------------------------
def apply_position_limits(qty: int, price: float, cash: float, symbol: str) -> int:
    """
    Apply hard position size limits to prevent over-concentration.

    Args:
        qty: Proposed quantity to buy
        price: Current price per share
        cash: Available cash
        symbol: Symbol being traded

    Returns:
        Adjusted quantity after limits
    """
    if qty <= 0 or price <= 0:
        return 0

    proposed_value = qty * price

    # Limit 1: Percentage of cash
    max_by_pct = int((cash * MAX_POSITION_SIZE_PCT) // price)

    # Limit 2: Absolute dollar amount (if configured)
    if MAX_POSITION_SIZE_DOLLARS is not None:
        max_by_dollars = int(MAX_POSITION_SIZE_DOLLARS // price)
        max_allowed = min(max_by_pct, max_by_dollars)
    else:
        max_allowed = max_by_pct

    # Return lesser of proposed and max allowed
    if qty > max_allowed:
        print(f"[RISK] {symbol}: Position size limited {qty} → {max_allowed} "
              f"(${proposed_value:.2f} → ${max_allowed * price:.2f})")
        return max_allowed

    return qty



# ---------------------------------------------------------
# Helper to build clean decision dicts
# ---------------------------------------------------------
def make_decision(action: str, qty: int, explain: str, **meta):
    d = {"action": action, "qty": int(qty), "explain": explain.strip()}
    d.update(meta)  # allow flags like recalc_all_in=True
    return d


def _core_buy_intent(preds: Dict[str, float], core_symbols: List[str]) -> List[str]:
    """Symbols that WANT to buy based on prob only (ignores cash/qty)."""
    return [s for s in core_symbols if preds.get(s, 0.0) >= BUY_THRESHOLD]


def _any_core_buy(decisions: Dict[str, dict], core_symbols: List[str]) -> bool:
    """True if any core symbol has a BUY decision."""
    return any((decisions.get(s) or {}).get("action") == "buy" for s in core_symbols)


def _force_spy_exit_if_core_buy(decisions: Dict[str, dict], spy_sym: str, explain_suffix: str = ""):
    """If core has a BUY, sell SPY if held; otherwise hold SPY."""
    spy_pm = PortfolioManager(spy_sym)
    try:
        spy_pm.refresh_live()
    except:
        pass
    spy_shares = float(spy_pm.data.get("shares", 0.0))


    if spy_shares > 0:
        decisions[spy_sym] = make_decision(
            "sell",
            int(spy_shares),
            f"{spy_sym}: SELL (rotate into core) — core BUY opportunity detected. {explain_suffix}".strip()
        )
    else:
        decisions[spy_sym] = make_decision(
            "hold",
            0,
            f"{spy_sym}: HOLD — no position, core BUY opportunity detected. {explain_suffix}".strip()
        )


# ---------------------------------------------------------
# Helper for weak market
# ---------------------------------------------------------
def _weak_market(symbols: List[str], preds: Dict[str, float]) -> bool:
    """Return True if enough of the non-SPY symbols are weak."""
    universe = [s for s in symbols if s != SPY_SYMBOL]
    if not universe:
        return False


    weak = [s for s in universe if preds.get(s, 1.0) <= WEAK_PROB_THRESHOLD]
    ratio = len(weak) / max(len(universe), 1)
    return ratio >= WEAK_RATIO_THRESHOLD



def _any_stock_trade(decisions: Dict[str, dict], symbols: List[str]) -> bool:
    """True if any non-SPY symbol has buy/sell decision."""
    for s in symbols:
        if s == SPY_SYMBOL:
            continue
        a = (decisions.get(s) or {}).get("action", "hold")
        if a in ("buy", "sell"):
            return True
    return False



# ---------------------------------------------------------
# Evaluate STOP-LOSS / TAKE-PROFIT (universal)
# ---------------------------------------------------------
def check_stop_tp(symbol, price, pm):
    symbol = str(symbol).upper().strip()


    shares = float(pm.data.get("shares", 0.0))
    if shares <= 0:
        return None


    entry_price = float(pm.data.get("avg_price", pm.data.get("last_price", 0.0)) or 0.0)
    if entry_price <= 0:
        return None


    # ---------------------------------------------------------
    # PDT context (only used to limit selling opened-today shares)
    # ---------------------------------------------------------
    pdt = None
    eq = 0.0
    dt = 0


    opened_today = 0.0
    sellable_overnight = shares  # default = all shares sellable


    if PDT_TIERING_ENABLED:
        try:
            pdt = get_pdt_status()  # your zero-arg version
        except Exception:
            pdt = None


        if pdt:
            try:
                eq = float(pdt.get("equity", 0) or 0)
                dt = int(pdt.get("daytrade_count", 0) or 0)
            except Exception:
                eq, dt = 0.0, 0


            # Only tier when PDT matters and you're near limit
            if eq < 25000 and dt >= 3:
                try:
                    opened_today = float(get_opened_today_qty(symbol) or 0.0)
                except Exception:
                    opened_today = 0.0


                opened_today = max(0.0, min(opened_today, shares))
                sellable_overnight = max(0.0, shares - opened_today)


    # Maintain max_price while holding
    mp = float(pm.data.get("max_price", entry_price) or entry_price)
    if price > mp:
        pm.data["max_price"] = price
        try:
            pm.save()
        except:
            pass
        mp = price


    # helpers
    def _loss_pct() -> float:
        # positive number when losing, e.g. 0.03 == -3%
        return max(0.0, 1.0 - (float(price) / float(entry_price)))


    def _pdt_tiered_sell(reason: str):
        """
        PDT-aware selling:
        - If near PDT limit: sell ONLY overnight shares (shares - opened_today)
        - If everything was opened today: block unless emergency stop triggers
        - If opened_today == 0: selling is safe (won't create a day-trade) -> sell all
        """
        near_pdt = bool(PDT_TIERING_ENABLED and pdt and eq < 25000 and dt >= 3)


        # Not near PDT limit => sell everything normally
        if not near_pdt:
            return make_decision("sell", int(shares), reason)


        # If we have any overnight shares, sell ONLY those (avoid day trade)
        if sellable_overnight > 0:
            # If opened_today==0 then sellable_overnight==shares, so this sells all anyway.
            return make_decision(
                "sell",
                int(sellable_overnight),
                reason + f" | PDT-tier: sold overnight={sellable_overnight:g}, blocked opened_today={opened_today:g}"
            )


        # At this point: sellable_overnight == 0 (everything opened today)
        loss = _loss_pct()


        # Emergency override => allow same-day exit (day-trade risk accepted)
        if PDT_EMERGENCY_STOP is not None and loss >= float(PDT_EMERGENCY_STOP):
            d = make_decision(
                "sell",
                int(shares),
                reason + f" | 🚨 PDT EMERGENCY stop (loss={loss:.2%}) allowing same-day exit"
            )
            d["pdt_emergency"] = True
            return d


        # Otherwise block
        return make_decision(
            "hold",
            0,
            f"{symbol}: STOP blocked by PDT tiering (opened_today={opened_today:g}, loss={loss:.2%})."
        )


    # 1) HARD STOP-LOSS
    if price <= entry_price * STOP_LOSS:
        return _pdt_tiered_sell(
            f"{symbol}: STOP-LOSS hit {price:.2f} <= {STOP_LOSS*100:.1f}% of entry {entry_price:.2f}"
        )


    # 2) TRAILING STOP (ONLY AFTER +5% PROFIT)
    if price >= entry_price * TRAIL_ACTIVATE:
        trail_level = mp * TRAIL_STOP
        if price <= trail_level:
            return _pdt_tiered_sell(
                (
                    f"{symbol}: TRAIL-STOP hit {price:.2f} <= {TRAIL_STOP*100:.1f}% of max {mp:.2f} "
                    f"(activated after +{(TRAIL_ACTIVATE-1)*100:.1f}% profit)"
                )
            )


    return None



# ---------------------------------------------------------
# Per-symbol trading logic (baseline)
# ---------------------------------------------------------
def should_trade(symbol: str, prob_up: float, total_symbols: int = 1,
                 concurrent_buys: int = 1, available_cash: float = None):


    pm = PortfolioManager(symbol)
    try:
        pm.refresh_live()
    except:
        pass


    shares = float(pm.data.get("shares", 0.0))
    cash = float(pm.data.get("cash", 0.0))
    price = fetch_latest_price(symbol)


    if price is None or price <= 0:
        return make_decision("hold", 0, f"{symbol}: no valid price.")


    if available_cash is not None:
        cash = float(available_cash)


    # Allocation logic
    if total_symbols == 1:
        max_invest = cash
    elif total_symbols == 2:
        max_invest = cash * RISK_FRACTION if concurrent_buys == 2 else cash
    else:
        max_invest = cash * RISK_FRACTION


    explain = (
        f"{symbol}: prob_up={prob_up:.3f} "
        f"(BUY≥{BUY_THRESHOLD}, SELL≤{SELL_THRESHOLD}, shares={shares:.4g}). "
    )


    # If already in a position, don't pyramid by default (prevents confusing "BUY but no cash")
    if shares > 0 and prob_up >= BUY_THRESHOLD:
        if prob_up < PYRAMID_THRESHOLD:
            return make_decision(
                "hold", 0,
                explain + f"HOLD — already in position (shares={shares:g}); "
                f"pyramiding requires prob≥{PYRAMID_THRESHOLD:.2f} (current={prob_up:.3f})."
            )
        
        # Pyramiding allowed - calculate quantity based on available cash
        # Use same allocation logic as new positions
        if total_symbols == 1:
            max_invest = cash
        elif total_symbols == 2:
            max_invest = cash * RISK_FRACTION if concurrent_buys == 2 else cash
        else:
            max_invest = cash * RISK_FRACTION
        
        affordable = int(cash // price)
        target = int(max_invest // price)
        qty = min(affordable, target)
        
        # Apply position limits
        qty = apply_position_limits(qty, price, cash, symbol)
        
        if qty > 0:
            return make_decision(
                "buy", qty,
                explain + f"PYRAMID BUY — adding to position (current={shares:g}, new={qty}, prob={prob_up:.3f})."
            )
        else:
            return make_decision(
                "hold", 0,
                explain + f"HOLD — pyramiding signal but insufficient cash (available=${cash:.2f})."
            )


    # BUY
    if prob_up >= BUY_THRESHOLD:
        affordable = int(cash // price)
        target = int(max_invest // price)
        qty = min(affordable, target)


        # NEW: Apply hard limits
        qty = apply_position_limits(qty, price, cash, symbol)


        if qty > 0:
            return make_decision("buy", qty, explain + f"BUY. qty={qty}")
        return make_decision("hold", 0, explain + "BUY signal but insufficient cash.")


    # SELL
    if prob_up <= SELL_THRESHOLD:
        if shares > 0:
            return make_decision("sell", int(shares), explain + "SELL. Clearing position.")
        else:
            return make_decision("hold", 0, explain + "SELL signal but no position.")


    # HOLD
    return make_decision("hold", 0, explain + "Within thresholds → HOLD.")


# ============================================================
# 🚀 MOMENTUM BREAKOUT DETECTION
# ============================================================
def check_momentum_breakout(sym: str, diagnostics: dict, preds: dict) -> tuple:
    """
    Detect strong momentum breakouts that model might underestimate.
    Returns: (force_buy: bool, reason: str)
    """
    import pandas as pd
    
    d = diagnostics.get(sym, {})
    prob = preds.get(sym, 0.0)
    
    # Get momentum metrics
    mom = d.get("intraday_mom")
    vol = d.get("intraday_vol")
    price = d.get("price")
    
    if not all([mom is not None, price]):
        return False, ""
    
    try:
        mom = float(mom)
        vol = float(vol) if vol else 0.0
        price = float(price)
        
        # Fetch 50-day MA for trend context
        df = fetch_historical_data(sym, period="3mo", interval="1d")
        if df is None or len(df) < 50:
            return False, ""
        
        # ✅ FIX: Extract scalar value from Series
        close_series = df['Close']
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]
        
        ma50_series = close_series.rolling(50).mean()
        ma50 = float(ma50_series.iloc[-1])  # ✅ Convert to scalar
        
        # Calculate how far above MA50
        above_ma = (price - ma50) / ma50
        
        # BREAKOUT CONDITIONS:
        # 1. Price >3% above 50-day MA (established uptrend)
        # 2. Intraday momentum >1% (strong continuation)
        # 3. Model at least neutral (prob >= 0.45)
        # 4. High volatility confirms real move (vol > 1.5%)
        
        if (
            above_ma > 0.02 and      # ✅ LOWERED: 2% above MA50 (was 3%)
            mom > 0.008 and          # ✅ LOWERED: 0.8% hourly momentum (was 1%)
            prob >= 0.40 and         # ✅ LOWERED: Model prob >=40% (was 45%)
            vol > 0.012):            # ✅ LOWERED: Vol >1.2% (was 1.5%)

            
            return True, (
                f"[MOMENTUM BREAKOUT] {sym}: "
                f"+{above_ma:.1%} above MA50, "
                f"mom_1h={mom:.2%}, "
                f"vol={vol:.2%}, "
                f"model_prob={prob:.1%}"
            )
            
    except Exception as e:
        print(f"[WARN] Momentum breakout check failed for {sym}: {e}")
    
    return False, ""


# ---------------------------------------------------------
# NVDA-priority coordinator
# ---------------------------------------------------------
def compute_strategy_decisions(
    predictions: Dict[str, float],
    symbols: List[str] = None,
    diagnostics: Dict[str, dict] = None,
):


    spy_sym = SPY_SYMBOL.upper()


    if symbols is None:
        symbols = list(predictions.keys())


    symbols = [s.upper() for s in symbols]
    preds = {k.upper(): v for k, v in predictions.items()}
    core_symbols = [s for s in symbols if s != spy_sym]


    # ---------------------------------------------------------
    # BUY GUARDRAIL HELPERS (use intraday diagnostics)
    # ---------------------------------------------------------
    diagnostics = diagnostics or {}



    def _diag(sym: str) -> dict:
        return diagnostics.get(sym.upper()) or {}


    def _block_buy_on_pullback(sym: str) -> bool:
        """
        Guardrail: avoid buying while short-term momentum is negative
        and intraday signal is weaker than daily.
        """
        d = _diag(sym)


        dp = d.get("daily_prob")
        ip = d.get("intraday_prob")
        mom = d.get("intraday_mom")   # e.g. -0.0060 == -0.60% over ~2h
        q = d.get("intraday_quality_score")


        try:
            dp = None if dp is None else float(dp)
            ip = None if ip is None else float(ip)
            mom = None if mom is None else float(mom)
            q = None if q is None else float(q)
        except Exception:
            return False


        # no intraday diagnostics => don't block
        if mom is None or ip is None or dp is None:
            return False


        # require reasonable intraday quality
        if q is not None and q < 0.60:
            return False


        # pullback + intraday weaker than daily
        if (mom <= -0.003) and (ip < dp):
            return True


        return False

    def _block_buy_on_weak_volume(sym: str) -> bool:
        """
        Guardrail: avoid full-size BUY when intraday volume is weak
        relative to its recent average.

        Returns True if volume is too weak for the current regime.
        """
        d = _diag(sym)
        vol_ratio = d.get("intraday_volume_ratio")
        regime = d.get("intraday_regime")  # "mom", "mr", or None

        try:
            vol_ratio = None if vol_ratio is None else float(vol_ratio)
        except Exception:
            vol_ratio = None

        if vol_ratio is None:
            # No volume data → don't block (fail open)
            return False

        # Regime-specific thresholds
        if regime == "mom":
            min_ratio = 1.20   # momentum needs volume expansion
        elif regime == "mr":
            min_ratio = 0.80   # mean-reversion can work in quieter conditions
        else:
            min_ratio = 1.00   # default for legacy intraday

        return vol_ratio < min_ratio


    # ---------------------------------------------------------
    # ABBV secondary logic helpers
    # ---------------------------------------------------------
    rs_margin = max(0.0, min(float(RS_MARGIN), 0.25))  # require ABBV to beat AAPL by 0.05 to switch (anti-churn)


    # ✅ MOVED: Define _is_buy() BEFORE pick_secondary_among_stocks()
    def _is_buy(sym: str) -> bool:
        """
        Core BUY predicate used for AAPL/ABBV/PLTR and other symbols.

        Returns True only if:
        - final_prob >= BUY_THRESHOLD
        - NOT blocked by momentum pullback guard
        - NOT blocked by weak volume guard
        """
        if preds.get(sym, 0.0) < BUY_THRESHOLD:
            return False

        if _block_buy_on_pullback(sym):
            return False

        if _block_buy_on_weak_volume(sym):
            return False

        return True


    # ✅ NOW pick_secondary_among_stocks() can use _is_buy()
    def pick_secondary_among_stocks() -> Optional[str]:
        """
        Choose best secondary stock when NVDA not buying.
        Candidates: AAPL, ABBV, PLTR (mutual exclusive).
        Returns highest probability stock above BUY_THRESHOLD.
        """
        candidates = {}


        for sym in ["AAPL", "ABBV", "PLTR"]:
            if sym in core_symbols:
                prob = preds.get(sym)
                if prob is not None and _is_buy(sym):  # ✅ Now visible!
                    candidates[sym] = prob


        if not candidates:
            return None


        # Return highest probability
        best_sym = max(candidates.keys(), key=lambda s: candidates[s])
        best_prob = candidates[best_sym]


        # Optional: margin requirement (prevent flipping on tiny differences)
        others = {s: p for s, p in candidates.items() if s != best_sym}
        if others:
            second_best = max(others.values())
            if best_prob < second_best + rs_margin:  # too close - keep incumbent
                # Check who we're holding
                for sym in [best_sym] + list(others.keys()):
                    # pms will be defined later in compute_strategy_decisions
                    # This is safe because pick_secondary_among_stocks() is only called
                    # after pms is populated
                    pass  # Will fix after pms is defined


        return best_sym


    # Default decisions
    decisions = {s: make_decision("hold", 0, f"{s}: default HOLD") for s in symbols}


    # ---------------------------------------------------------
    # Fetch live state + prices (for stop/tp + funding calcs)
    # ---------------------------------------------------------

    # ✅ Fetch account state ONCE
    account_cache.invalidate()  # Fresh data for this strategy cycle
    account_state = account_cache.get_account()


    pms = {}
    prices = {}
    for sym in symbols:
        pm = PortfolioManager(sym)
        try:
            pm.refresh_live()
        except:
            pass


        pms[sym] = pm
        d = diagnostics.get(sym.upper()) or {}
        prices[sym] = float(d.get("price") or 0.0) or (fetch_latest_price(sym) or 0.0)


    # Ensure SPY state exists even if not in symbols (safe)
    if spy_sym not in pms:
        pm = PortfolioManager(spy_sym)
        try:
            pm.refresh_live()
        except:
            pass
        pms[spy_sym] = pm
        prices[spy_sym] = fetch_latest_price(spy_sym) or 0.0


    # ✅ NOW fix pick_secondary_among_stocks() to use pms
    # We need to redefine it here after pms is available
    def pick_secondary_among_stocks_fixed() -> Optional[str]:
        """
        Choose best secondary stock when NVDA not buying.
        Candidates: AAPL, ABBV, PLTR (mutual exclusive).
        Returns highest probability stock above BUY_THRESHOLD.
        """
        candidates = {}


        for sym in ["AAPL", "ABBV", "PLTR"]:
            if sym in core_symbols:
                prob = preds.get(sym)
                if prob is not None and _is_buy(sym):
                    candidates[sym] = prob


        if not candidates:
            return None


        # Return highest probability
        best_sym = max(candidates.keys(), key=lambda s: candidates[s])
        best_prob = candidates[best_sym]


        # Optional: margin requirement (prevent flipping on tiny differences)
        others = {s: p for s, p in candidates.items() if s != best_sym}
        if others:
            second_best = max(others.values())
            if best_prob < second_best + rs_margin:  # too close - keep incumbent
                for sym in [best_sym] + list(others.keys()):
                    if float(pms[sym].data.get("shares", 0)) > 0:
                        return sym  # keep what we hold


        return best_sym

    # Use the fixed version
    pick_secondary_among_stocks = pick_secondary_among_stocks_fixed


    def spy_shares() -> float:
        return float(pms[spy_sym].data.get("shares", 0.0))


    def spy_price() -> float:
        return float(prices.get(spy_sym, 0.0) or 0.0)


    # ---------------------------------------------------------
    # 1) STOP-LOSS / TAKE-PROFIT always first (including SPY)
    # ---------------------------------------------------------
    sl_tp_decisions = {}
    for sym in symbols:
        price = prices.get(sym, 0.0) or 0.0
        if price > 0:
            d = check_stop_tp(sym, price, pms[sym])
            if d:
                sl_tp_decisions[sym] = d


    if sl_tp_decisions:
        for sym in symbols:
            if sym in sl_tp_decisions:
                decisions[sym] = sl_tp_decisions[sym]
            else:
                decisions[sym] = make_decision("hold", 0, f"{sym}: HOLD during SL/TP event.")
        return decisions


    # ---------------------------------------------------------
    # 2) SPY candidate (ONLY for entering SPY on weak market)
    # ---------------------------------------------------------
    spy_candidate = None
    spy_prob = preds.get(spy_sym)


    if spy_prob is not None:
        market_is_weak = _weak_market(symbols, preds)  # your helper excludes SPY
        if market_is_weak:
            sh = spy_shares()
            cash = float(pms[spy_sym].data.get("cash", 0.0))
            px = spy_price()


            if px > 0:
                if spy_prob >= SPY_ENTRY_THRESHOLD:
                    qty = int((cash * SPY_RISK_FRACTION) // px)
                    spy_candidate = make_decision(
                        "buy",
                        max(qty, 0),
                        f"{spy_sym}: SPY fallback BUY — market weak and spy_prob={spy_prob:.3f} ≥ {SPY_ENTRY_THRESHOLD}"
                    )
                elif spy_prob <= SPY_EXIT_THRESHOLD and sh > 0:
                    spy_candidate = make_decision(
                        "sell",
                        int(sh),
                        f"{spy_sym}: SPY fallback SELL — spy_prob={spy_prob:.3f} ≤ {SPY_EXIT_THRESHOLD}"
                    )
                else:
                    spy_candidate = make_decision(
                        "hold", 0,
                        f"{spy_sym}: SPY fallback HOLD — spy_prob={spy_prob:.3f}"
                    )


    # ---------------------------------------------------------
    # 3) GLOBAL SELL (core only)
    # ---------------------------------------------------------
    valid_core_preds = [preds[s] for s in core_symbols if s in preds]
    if valid_core_preds and all(p <= SELL_THRESHOLD for p in valid_core_preds):
        for sym in symbols:
            sh = float(pms[sym].data.get("shares", 0.0))
            if sh > 0:
                decisions[sym] = make_decision("sell", int(sh), f"{sym}: Global SELL — liquidating.")
            else:
                decisions[sym] = make_decision("hold", 0, f"{sym}: Global SELL — no position.")
        return decisions


    # ---------------------------------------------------------
    # 4) Core strategy with NVDA priority (core only)
    # ---------------------------------------------------------
    if "NVDA" not in core_symbols:
        concurrent_buys = sum(1 for s in core_symbols if preds.get(s, 0.0) >= BUY_THRESHOLD)
        for sym in core_symbols:
            decisions[sym] = should_trade(
                sym, preds.get(sym, 0.0),
                total_symbols=len(core_symbols),
                concurrent_buys=concurrent_buys
            )
    else:
        def live_shares(sym): return float(pms[sym].data.get("shares", 0.0))


        concurrent_buys = sum(1 for s in core_symbols if preds.get(s, 0.0) >= BUY_THRESHOLD)


        nvda_prob = preds.get("NVDA", 0.0)
        nvda_base = should_trade("NVDA", nvda_prob, len(core_symbols), concurrent_buys)
        nvda_action = nvda_base["action"]


        # ---- NVDA BUY priority: sell other core positions (funding) + plan big buy
        if nvda_action == "buy":
            sim_cash = float(pms[core_symbols[0]].data.get("cash", 0.0))
            nvda_px = float(prices.get("NVDA", 0.0) or 0.0)
            if nvda_px <= 0:
                return decisions


            # ---------------------------------------------------------
            # NVDA ROTATION RULE:
            # If NVDA is BUY and AAPL is NOT BUY (hold/sell), then sell AAPL (if held)
            # to fund NVDA.
            # ---------------------------------------------------------
            aapl_prob = preds.get("AAPL", 0.0)
            aapl_sig = should_trade("AAPL", aapl_prob, len(core_symbols), concurrent_buys)
            aapl_action = (aapl_sig.get("action") or "hold").lower()


            # fund with other core positions (excluding NVDA)
            for sym in core_symbols:
                if sym == "NVDA":
                    continue

                sh = live_shares(sym)
                px = float(prices.get(sym, 0.0) or 0.0)
                if sh <= 0 or px <= 0:
                    continue

                # ✅ PDT CHECK: Don't sell if opened today
                try:
                    opened_today_qty = float(get_opened_today_qty(sym) or 0.0)
                    if opened_today_qty >= sh:  # Entire position opened today
                        print(f"[PDT BLOCK] Cannot sell {sym} to fund NVDA - entire position opened today ({sh} shares)")
                        continue  # Skip this symbol, don't create SELL decision
                except Exception as e:
                    print(f"[WARN] PDT check failed for {sym}: {e}")
                    # Fail safe - don't sell if we can't verify

                sim_cash += sh * px
                decisions[sym] = make_decision(
                    "sell",
                    int(sh),
                    f"{sym}: Sold to fund NVDA BUY (NVDA priority)."
                )



                # Only force-sell AAPL if NVDA BUY and AAPL is HOLD/SELL (not BUY)
                if sym == "AAPL":
                    if aapl_action != "buy":
                        sim_cash += sh * px
                        decisions["AAPL"] = make_decision(
                            "sell",
                            int(sh),
                            f"AAPL: Rotated out to fund NVDA BUY (AAPL={aapl_action.upper()}, NVDA=BUY)."
                        )
                    # else: AAPL is also BUY -> keep existing behavior (don't force sell it)
                    continue


                # For other symbols (if you add more later), keep the old "sell to fund NVDA"
                sim_cash += sh * px
                decisions[sym] = make_decision("sell", int(sh), f"{sym}: Sold to fund NVDA priority buy.")

            max_shares = int(sim_cash // nvda_px)


            # NEW: Apply limits
            max_shares = apply_position_limits(max_shares, nvda_px, sim_cash, "NVDA")


            if max_shares >= 1:
                decisions["NVDA"] = make_decision(
                    "buy",
                    max_shares,
                    f"NVDA BUY priority — capital ${sim_cash:.2f}, qty={max_shares}"
                )
            else:
                decisions["NVDA"] = make_decision("hold", 0, "NVDA BUY priority but insufficient capital.")


            # NOTE: do NOT apply SPY here; final enforcement below will handle it.
        elif nvda_action == "sell":
            sh = live_shares("NVDA")
            if sh > 0:
                decisions["NVDA"] = make_decision("sell", int(sh), "NVDA SELL signal.")
            else:
                decisions["NVDA"] = make_decision("hold", 0, "NVDA SELL but no position.")


            # rotate into strongest other BUY
            buyers = [s for s in core_symbols if s != "NVDA" and preds.get(s, 0.0) >= BUY_THRESHOLD]
            if buyers:
                strongest = max(buyers, key=lambda x: preds.get(x, 0.0))
                cash_now = float(pms[core_symbols[0]].data.get("cash", 0.0)) + sh * float(prices.get("NVDA", 0.0) or 0.0)
                tgt_px = float(prices.get(strongest, 0.0) or 0.0)
                qty = int(cash_now // tgt_px) if tgt_px > 0 else 0
                if qty >= 1:
                    decisions[strongest] = make_decision("buy", qty, f"{strongest}: Rotated from NVDA sell, qty={qty}")
        else:
            # NVDA HOLD -> allow at most ONE secondary BUY: AAPL vs ABBV vs PLTR
            secondary = pick_secondary_among_stocks()


            for sym in core_symbols:
                d0 = should_trade(
                    sym, preds.get(sym, 0.0),
                    total_symbols=len(core_symbols),
                    concurrent_buys=concurrent_buys
                )


                # suppress BUY for the non-selected secondary candidate
                if d0.get("action") == "buy" and secondary is not None and sym in ("AAPL", "ABBV", "PLTR") and sym != secondary:
                    d0 = make_decision("hold", 0, f"{sym}: BUY suppressed (secondary={secondary}).")


                decisions[sym] = d0


            # If ABBV chosen AND AAPL is not BUY AND we hold AAPL -> sell AAPL to fund ABBV
            if secondary == "ABBV" and "AAPL" in core_symbols and "ABBV" in core_symbols:
                aapl_action = (decisions.get("AAPL") or {}).get("action", "hold")
                aapl_sh = float(pms["AAPL"].data.get("shares", 0.0) or 0.0)


                if aapl_sh > 0 and aapl_action in ("hold", "sell"):
                    # ✅ PDT CHECK
                    try:
                        opened_today_qty = float(get_opened_today_qty("AAPL") or 0.0)
                        if opened_today_qty >= aapl_sh:
                            print(f"[PDT BLOCK] Cannot sell AAPL to fund ABBV - opened today")
                        else:
                            decisions["AAPL"] = make_decision("sell", int(aapl_sh), "AAPL: Sold to fund ABBV BUY (rotation).")
                    except Exception as e:
                        print(f"[WARN] PDT check failed for AAPL: {e}")



   # ---------------------------------------------------------
    # 4.5) BUY CONFIRMATION GUARDRAIL
    # ---------------------------------------------------------
    def _safe_f(x, default=None):
        try:
            return float(x)
        except Exception:
            return default


    for sym in core_symbols:
        d = decisions.get(sym) or {}
        if d.get("action") == "buy":
            if _block_buy_on_pullback(sym):
                dd = _diag(sym)
                mom = _safe_f(dd.get("intraday_mom"))
                ip  = _safe_f(dd.get("intraday_prob"))
                dp  = _safe_f(dd.get("daily_prob"))


                mom_str = "NA" if mom is None else f"{mom:.2%}"
                ip_str  = "NA" if ip is None else f"{ip:.3f}"
                dp_str  = "NA" if dp is None else f"{dp:.3f}"


                decisions[sym] = make_decision(
                    "hold",
                    0,
                    f"{sym}: BUY blocked by pullback guardrail (mom={mom_str}, ip={ip_str} < dp={dp_str})."
                )

    # ---------------------------------------------------------
    # 4.7) ROTATION: If NVDA wants BUY and AAPL is not BUY, sell AAPL to fund NVDA
    # ---------------------------------------------------------
    if "NVDA" in core_symbols and "AAPL" in core_symbols:
        nvda_d = decisions.get("NVDA") or {}
        aapl_d = decisions.get("AAPL") or {}


        nvda_wants_buy = (nvda_d.get("action") == "buy") or (preds.get("NVDA", 0.0) >= BUY_THRESHOLD)
        aapl_is_not_buy = (aapl_d.get("action") in ("hold", "sell")) and (preds.get("AAPL", 0.0) < BUY_THRESHOLD + 0.05) 


        if nvda_wants_buy and aapl_is_not_buy:
            aapl_sh = float(pms["AAPL"].data.get("shares", 0.0) or 0.0)
            if aapl_sh > 0:
                # ✅ PDT CHECK
                try:
                    opened_today_qty = float(get_opened_today_qty("AAPL") or 0.0)
                    if opened_today_qty >= aapl_sh:
                        print(f"[PDT BLOCK] Cannot rotate AAPL to fund NVDA - opened today")
                    else:
                        decisions["AAPL"] = make_decision(
                            "sell",
                            int(aapl_sh),
                            "AAPL: Sold to fund NVDA rotation buy."
                        )
                except Exception as e:
                    print(f"[WARN] PDT check failed for AAPL rotation: {e}")



            # mark NVDA as priority buy (main.py will recalc after sells)
            decisions["NVDA"] = make_decision(
                "buy", 1,
                "NVDA PRIORITY BUY — recalc all-in after sells (AAPL rotation + SPY liquidation if any).",
                recalc_all_in=True,
                priority_rank=1,
            )


    # ---------------------------------------------------------
    # 5) FINAL ENFORCEMENT: if NVDA/AAPL/ABBV/PLTR has BUY INTENT => SELL SPY + recalc flags
    # (prob-based intent, not cash-based)
    # ---------------------------------------------------------
    wanted = []
    if "NVDA" in core_symbols and preds.get("NVDA", 0.0) >= BUY_THRESHOLD and not _block_buy_on_pullback("NVDA"):
        wanted.append("NVDA")
    if "AAPL" in core_symbols and preds.get("AAPL", 0.0) >= BUY_THRESHOLD and not _block_buy_on_pullback("AAPL"):
        wanted.append("AAPL")
    if "ABBV" in core_symbols and preds.get("ABBV", 0.0) >= BUY_THRESHOLD and not _block_buy_on_pullback("ABBV"):
        wanted.append("ABBV")
    if "PLTR" in core_symbols and preds.get("PLTR", 0.0) >= BUY_THRESHOLD and not _block_buy_on_pullback("PLTR"):
        wanted.append("PLTR")


    if wanted:
        sh_spy = float(spy_shares() or 0.0)


        # --------------------------------------------
        # Funding sanity: only do "recalc after sells"
        # if we can actually raise cash.
        # --------------------------------------------
        # Anything we can liquidate this cycle?
        has_other_core_positions = any(
            float(pms[s].data.get("shares", 0.0) or 0.0) > 0.0
            for s in core_symbols
            if s not in ("NVDA", "AAPL", "ABBV", "PLTR")   # exclude targets; optional but recommended
        )


        has_funding = (sh_spy > 0.0) or has_other_core_positions


        if not has_funding:
            # Nothing to sell -> don't emit fake "priority buy recalc" intents
            # Just keep the earlier computed decisions (including guardrail HOLDs).
            return decisions


        # If we DO have SPY, sell it to fund core
        if sh_spy > 0.0:
            decisions[spy_sym] = make_decision(
                "sell",
                int(sh_spy),
                f"{spy_sym}: SELL (rotate into core) — core BUY intent: {', '.join(wanted)}."
            )


        # --------------------------------------------
        # mark priority buys for main.py to recalc AFTER sells
        # --------------------------------------------
        if "NVDA" in wanted:
            decisions["NVDA"] = make_decision(
                "buy", 1,
                "NVDA PRIORITY BUY — recalc all-in after sells (SPY liquidation).",
                recalc_all_in=True,
                priority_rank=1,
            )


            # choose best secondary among AAPL, ABBV, PLTR
            secondary = None
            secondary_candidates = {}
            for sym in ["AAPL", "ABBV", "PLTR"]:
                if sym in wanted:
                    secondary_candidates[sym] = preds.get(sym, 0.0)

            if secondary_candidates:
                # Pick highest probability with margin
                sorted_candidates = sorted(secondary_candidates.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_candidates) == 1:
                    secondary = sorted_candidates[0][0]
                else:
                    best = sorted_candidates[0]
                    second_best = sorted_candidates[1]
                    if best[1] > second_best[1] + rs_margin:
                        secondary = best[0]
                    else:
                        # Too close - keep incumbent
                        for sym, _ in sorted_candidates:
                            if float(pms[sym].data.get("shares", 0)) > 0:
                                secondary = sym
                                break
                        if secondary is None:
                            secondary = best[0]


            if secondary:
                decisions[secondary] = make_decision(
                    "buy", 1,
                    f"{secondary} BUY intent — secondary to NVDA (after sells).",
                    recalc_after_sells=True,
                    priority_rank=2,
                )


        else:
            # NVDA not wanted -> pick best of AAPL/ABBV/PLTR if any
            secondary = None
            secondary_candidates = {}
            for sym in ["AAPL", "ABBV", "PLTR"]:
                if sym in wanted:
                    secondary_candidates[sym] = preds.get(sym, 0.0)

            if secondary_candidates:
                sorted_candidates = sorted(secondary_candidates.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_candidates) == 1:
                    secondary = sorted_candidates[0][0]
                else:
                    best = sorted_candidates[0]
                    second_best = sorted_candidates[1]
                    if best[1] > second_best[1] + rs_margin:
                        secondary = best[0]
                    else:
                        for sym, _ in sorted_candidates:
                            if float(pms[sym].data.get("shares", 0)) > 0:
                                secondary = sym
                                break
                        if secondary is None:
                            secondary = best[0]


            if secondary:
                decisions[secondary] = make_decision(
                    "buy", 1,
                    f"{secondary} PRIORITY BUY — recalc all-in after sells (SPY liquidation).",
                    recalc_all_in=True,
                    priority_rank=1,
                )


        # When core wants to buy, we do NOT allow SPY buy this cycle
        return decisions


    # ---------------------------------------------------------
    # 6) If no core buy intent, allow SPY candidate (entry/exit) with mutual exclusive rules
    # ---------------------------------------------------------
    if spy_candidate is not None:
        if (not SPY_MUTUAL_EXCLUSIVE) or (not _any_stock_trade(decisions, core_symbols)):
            decisions[spy_sym] = spy_candidate
        else:
            decisions[spy_sym] = make_decision("hold", 0, f"{spy_sym}: Mutual-exclusive → skipping SPY this cycle.")

    
        # ============================================================
    # 🚀 MOMENTUM BREAKOUT OVERRIDE (before dip-buy)
    # ============================================================
    for sym in core_symbols:
        force_buy, reason = check_momentum_breakout(sym, diagnostics, preds)
        
        if force_buy:
            # Check if we can afford it
            pm = pms[sym]
            shares = float(pm.data.get("shares", 0.0))
            
            if shares > 0:
                # Already holding - keep it
                print(reason + " → HOLDING existing position")
                continue
            
            cash = float(account_state.get("cash", 0.0))
            price = float(prices.get(sym, 0.0))
            
            if price <= 0:
                continue
            
            # Allocate 80% of cash for breakout (aggressive)
            buy_qty = int((cash * 0.80) // price)
            buy_qty = apply_position_limits(buy_qty, price, cash, sym)
            
            if buy_qty > 0:
                decisions[sym] = make_decision(
                    "buy",
                    buy_qty,
                    reason,
                    momentum_override=True,
                    priority_rank=1
                )
                print(f"✅ {reason} → BUY {buy_qty} shares")
    
    # ============================================================
    # 🚨 EXTREME MOMENTUM OVERRIDE (>1.5% hourly move)
    # ============================================================
    for sym in core_symbols:
        d = diagnostics.get(sym, {})
        mom = d.get("intraday_mom")
        prob = preds.get(sym, 0.0)
        pm = pms[sym]
        shares = float(pm.data.get("shares", 0.0))
        
        if mom is not None and float(mom) > 0.015:  # >1.5% hourly momentum
            # If not holding and model is negative, override to neutral/buy
            if shares <= 0 and prob < 0.50:
                print(f"⚡ [EXTREME MOMENTUM] {sym} mom={float(mom):.2%} overriding prob {prob:.2%} → treating as neutral")
                # Don't force sell on extreme upward momentum
                if decisions.get(sym, {}).get("action") == "sell":
                    decisions[sym] = make_decision(
                        "hold", 0,
                        f"{sym}: SELL blocked by extreme upward momentum (mom={float(mom):.2%})"
                    )


    # ============================================================
    # 🔥 DIP-BUY OVERRIDE
    # ============================================================
    if DIP_BUY_ENABLED:
        for sym, decision in decisions.items():
            if decision.get("action") == "buy":
                prob = preds.get(sym, 0.0)

                if prob >= DIP_BUY_MIN_PROB:
                    if detect_afterhours_dip(sym, threshold=DIP_BUY_THRESHOLD):

                        print(f"🚀 [DIP-BUY] {sym} prob={prob:.2f} + dip → 100% capital override!")

                        try:
                            price = prices.get(sym, 0.0) or fetch_latest_price(sym) or 0.0
                            if price > 0:
                                pm = pms.get(sym)
                                if pm:
                                    cash = float(pm.data.get("cash", 0.0))
                                    new_qty = int(cash * 0.98 / price)

                                    if new_qty > decision.get("qty", 0):
                                        old_qty = decision.get("qty", 0)
                                        decision["qty"] = new_qty
                                        decision["explain"] = f"{decision.get('explain', '')} [DIP-BUY 100%]"
                                        print(f"   → {old_qty} → {new_qty} shares")
                        except Exception as e:
                            print(f"[ERROR] Dip-buy: {e}")


    return decisions
