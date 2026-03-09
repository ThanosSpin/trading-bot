#!/usr/bin/env python
"""
Analyze trading performance from trade logs.
Usage: python analyze_trades.py [--days 7] [--symbol NVDA]
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import glob
import os

# ── Risk config (mirror config.py) ──────────────────────────────────────────
STOP_LOSS       = 0.95   # 5% below buy price
TRAIL_STOP      = 0.96   # 4% trailing stop
TRAIL_ACTIVATE  = 1.05   # trailing activates after +5%


def load_trade_logs(symbol=None, days=None):
    """Load trade logs from CSV files."""
    search_paths = [".", "data", "logs", "../logs", "./logs/trades"]
    files = []

    for path in search_paths:
        if not os.path.exists(path):
            continue
        if symbol:
            pattern = os.path.join(path, f"trades_{symbol.upper()}.csv")
            # ✅ Also search paper trades for this symbol
            pattern_paper = os.path.join(path, f"paper_trades_{symbol.upper()}.csv")
            files.extend(glob.glob(pattern_paper))
        else:
            pattern = os.path.join(path, "trades_*.csv")
            # ✅ Also pick up all paper trade files
            pattern_paper = os.path.join(path, "paper_trades_*.csv")
            files.extend(glob.glob(pattern_paper))

        files.extend(glob.glob(pattern))

    files = list(set(files))

    if not files:
        print("❌ No trade log files found (trades_*.csv)")
        print("\nSearched in:")
        for path in search_paths:
            print(f"  - {path}")
        print("\n💡 Tip: Make sure your portfolio.log_trade() creates trades_*.csv files")
        return None

    print(f"✅ Found {len(files)} trade log file(s):")
    for f in files:
        print(f"   - {f}")

    dfs = []
    for file in files:
        try:
            df = pd.read_csv(file)

            timestamp_col = None
            for col in ["timestamp", "date", "time", "datetime"]:
                if col in df.columns:
                    timestamp_col = col
                    break

            if timestamp_col:
                try:
                    df["timestamp"] = pd.to_datetime(df[timestamp_col], format="ISO8601", utc=True)
                except Exception:
                    try:
                        df["timestamp"] = pd.to_datetime(df[timestamp_col], format="mixed", utc=True)
                    except Exception:
                        df["timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce")
                        if df["timestamp"].dt.tz is None:
                            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

            if "symbol" not in df.columns:
                sym = os.path.basename(file).replace("trades_", "").replace(".csv", "")
                df["symbol"] = sym.upper()

            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {file}: {e}")

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)

    if days and "timestamp" in df.columns:
        cutoff = pd.Timestamp.now(tz="UTC") - timedelta(days=days)
        df = df[df["timestamp"] >= cutoff]

    return df


# ── FIFO round-trip matching ─────────────────────────────────────────────────
def calculate_round_trips(df, price_col="price", qty_col="qty"):
    """
    Match buys to sells using FIFO.
    Returns (round_trips_df, open_positions_list).
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    buy_queue   = []   # list of dicts: {price, qty, time}
    round_trips = []

    for _, row in df.iterrows():
        action = str(row["action"]).upper()
        qty    = float(row[qty_col])
        price  = float(row[price_col])
        ts     = row["timestamp"]

        if action == "BUY":
            buy_queue.append({"price": price, "qty": qty, "time": ts})

        elif action == "SELL":
            remaining = qty
            while remaining > 0 and buy_queue:
                b        = buy_queue[0]
                matched  = min(b["qty"], remaining)
                pnl      = (price - b["price"]) * matched
                pnl_pct  = (price / b["price"]) - 1.0

                # Classify exit reason
                if pnl_pct <= -(1 - STOP_LOSS):          # hit hard stop ~5%
                    exit_reason = "stop_loss"
                elif pnl_pct >= (TRAIL_ACTIVATE - 1) and pnl_pct <= -(1 - TRAIL_STOP):
                    exit_reason = "trail_stop"
                elif pnl > 0:
                    exit_reason = "signal_profit"
                else:
                    exit_reason = "signal_loss"

                try:
                    hold_h = (ts - b["time"]).total_seconds() / 3600
                except Exception:
                    hold_h = 0.0

                round_trips.append({
                    "symbol":      row.get("symbol", ""),
                    "buy_time":    b["time"],
                    "sell_time":   ts,
                    "buy_price":   b["price"],
                    "sell_price":  price,
                    "qty":         matched,
                    "pnl":         round(pnl, 4),
                    "pnl_pct":     round(pnl_pct * 100, 3),
                    "hold_hours":  round(hold_h, 2),
                    "win":         pnl > 0,
                    "exit_reason": exit_reason,
                })

                b["qty"]  -= matched
                remaining -= matched
                if b["qty"] <= 0:
                    buy_queue.pop(0)

    rt_df     = pd.DataFrame(round_trips)
    open_pos  = [b for b in buy_queue if b["qty"] > 0]
    return rt_df, open_pos


def calculate_metrics(df):
    """Calculate trading performance metrics using proper FIFO round-trips."""
    if df is None or df.empty:
        return None

    metrics = {}

    if "action" in df.columns:
        trades = df[df["action"].str.upper().isin(["BUY", "SELL"])].copy()
    else:
        trades = df.copy()

    metrics["total_trades"] = len(trades)

    if metrics["total_trades"] == 0:
        print("⚠️ No trades found in the specified period")
        return metrics

    # Detect column names
    price_col = next((c for c in df.columns if "price" in c.lower()), None)
    qty_col   = next((c for c in df.columns if c.lower() in ["qty", "quantity", "shares", "amount"]), None)

    metrics["total_value_traded"] = 0
    if price_col and qty_col:
        trades[price_col] = pd.to_numeric(trades[price_col], errors="coerce")
        trades[qty_col]   = pd.to_numeric(trades[qty_col],   errors="coerce")
        trades = trades.dropna(subset=[price_col, qty_col])
        metrics["total_value_traded"] = (
            trades[price_col].astype(float) * trades[qty_col].astype(float)
        ).sum()

    # Count buys/sells
    if "action" in trades.columns:
        ac = trades["action"].str.upper().value_counts()
        metrics["buys"]  = int(ac.get("BUY",  0))
        metrics["sells"] = int(ac.get("SELL", 0))

    if "action" in df.columns:
        metrics["reconciliations"] = int(
            len(df[df["action"].str.upper() == "RECONCILE"])
        )

    if "notes" in df.columns:
        metrics["contradictions_logged"] = int(
            df["notes"].astype(str).str.contains("contradiction", case=False, na=False).sum()
        )

    # ── Per-symbol FIFO round-trip analysis ──────────────────────────────────
    if not (price_col and qty_col):
        metrics["total_realized_pnl"] = 0
        return metrics

    pnl_by_symbol  = {}
    all_round_trips = []

    for symbol in trades["symbol"].unique():
        sym_df = trades[trades["symbol"] == symbol].copy()
        if "timestamp" in sym_df.columns:
            sym_df = sym_df.sort_values("timestamp")

        rt_df, open_pos = calculate_round_trips(sym_df, price_col=price_col, qty_col=qty_col)
        all_round_trips.append(rt_df)

        realized_pnl = float(rt_df["pnl"].sum()) if not rt_df.empty else 0.0
        open_shares  = sum(b["qty"] for b in open_pos)
        open_avg     = (
            sum(b["price"] * b["qty"] for b in open_pos) / open_shares
            if open_shares > 0 else 0.0
        )

        # Per-round-trip stats
        wins       = int(rt_df["win"].sum())       if not rt_df.empty else 0
        losses     = len(rt_df) - wins             if not rt_df.empty else 0
        stop_hits  = int((rt_df["exit_reason"] == "stop_loss").sum()) if not rt_df.empty else 0
        avg_win    = float(rt_df[rt_df["win"]]["pnl"].mean())        if not rt_df.empty and rt_df["win"].any()  else 0.0
        avg_loss   = float(rt_df[~rt_df["win"]]["pnl"].mean())       if not rt_df.empty and (~rt_df["win"]).any() else 0.0
        avg_hold   = float(rt_df["hold_hours"].mean())               if not rt_df.empty else 0.0
        total_rt   = len(rt_df)
        win_rate   = wins / total_rt * 100 if total_rt > 0 else 0.0
        rr_ratio   = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        pnl_by_symbol[symbol] = {
            "realized_pnl":  realized_pnl,
            "position":      open_shares,
            "avg_price":     open_avg,
            "round_trips":   total_rt,
            "wins":          wins,
            "losses":        losses,
            "win_rate":      win_rate,
            "avg_win":       avg_win,
            "avg_loss":      avg_loss,
            "avg_hold_h":    avg_hold,
            "rr_ratio":      rr_ratio,
            "stop_hits":     stop_hits,
            "rt_df":         rt_df,
            "open_pos":      open_pos,
        }

    # Combine all round trips
    if all_round_trips:
        valid_rts = [r for r in all_round_trips if not r.empty]
        combined_rt = pd.concat(valid_rts, ignore_index=True) if valid_rts else pd.DataFrame()
    else:
        combined_rt = pd.DataFrame()

    metrics["pnl_by_symbol"]      = pnl_by_symbol
    metrics["total_realized_pnl"] = sum(s["realized_pnl"] for s in pnl_by_symbol.values())
    metrics["combined_rt"]        = combined_rt

    # Overall win rate across ALL round-trips (not per symbol)
    if not combined_rt.empty:
        total_rt   = len(combined_rt)
        total_wins = int(combined_rt["win"].sum())
        metrics["win_rate"]      = total_wins / total_rt * 100 if total_rt > 0 else 0.0
        metrics["total_rt"]      = total_rt
        metrics["total_wins"]    = total_wins
        metrics["total_losses"]  = total_rt - total_wins
        metrics["avg_win"]       = float(combined_rt[combined_rt["win"]]["pnl"].mean())  if combined_rt["win"].any()  else 0.0
        metrics["avg_loss"]      = float(combined_rt[~combined_rt["win"]]["pnl"].mean()) if (~combined_rt["win"]).any() else 0.0
        metrics["avg_hold_h"]    = float(combined_rt["hold_hours"].mean())
        metrics["stop_hits"]     = int((combined_rt["exit_reason"] == "stop_loss").sum())
        rr = abs(metrics["avg_win"] / metrics["avg_loss"]) if metrics["avg_loss"] != 0 else float("inf")
        metrics["rr_ratio"]      = rr
    else:
        metrics["win_rate"]     = 0.0
        metrics["total_rt"]     = 0
        metrics["total_wins"]   = 0
        metrics["total_losses"] = 0

    return metrics


def print_report(metrics, days=None):
    """Print formatted performance report."""
    print("\n" + "=" * 70)
    print("📊 TRADING PERFORMANCE REPORT")
    if days:
        print(f"Period: Last {days} days")
    print("=" * 70)

    if not metrics or metrics.get("total_trades", 0) == 0:
        print("\n⚠️ No trades to analyze")
        print("=" * 70 + "\n")
        return

    print(f"\n📈 TRADE STATISTICS:")
    print(f"  Total Rows Loaded : {metrics.get('total_trades', 0)}")
    print(f"  Buys              : {metrics.get('buys', 0)}")
    print(f"  Sells             : {metrics.get('sells', 0)}")
    print(f"  Completed R/Trips : {metrics.get('total_rt', 0)}")

    if metrics.get("reconciliations", 0) > 0:
        print(f"  Reconciliations   : {metrics['reconciliations']}")
    if metrics.get("contradictions_logged", 0) > 0:
        print(f"  Contradictions    : {metrics['contradictions_logged']}")

    print(f"\n💰 PROFIT & LOSS:")
    pnl = metrics.get("total_realized_pnl", 0)
    print(f"  Total Realized P&L  : ${pnl:+,.2f}")

    if "pnl_by_symbol" in metrics:
        print(f"\n  By Symbol:")
        for symbol, data in sorted(metrics["pnl_by_symbol"].items()):
            p         = data["realized_pnl"]
            pos       = data["position"]
            avg       = data["avg_price"]
            wr        = data["win_rate"]
            rt        = data["round_trips"]
            stop_hits = data["stop_hits"]
            rr        = data["rr_ratio"]
            avg_h     = data["avg_hold_h"]
            p_str     = f"${p:+,.2f}"
            stop_str  = f"  🛑 {stop_hits} stop-loss hit(s)" if stop_hits > 0 else ""
            print(f"    {symbol:6s}: {p_str:>10s}  |  {rt} R/Trips  |  WR {wr:.0f}%  |  R/R {rr:.2f}x  |  AvgHold {avg_h:.1f}h{stop_str}")
            print(f"           Avg Win ${data['avg_win']:+.2f}  |  Avg Loss ${data['avg_loss']:+.2f}  |  Open: {pos:.0f} @ ${avg:.2f}")

    print(f"\n  ── Overall ──────────────────────────────────────────")
    total_rt    = metrics.get("total_rt", 0)
    total_wins  = metrics.get("total_wins", 0)
    total_losses= metrics.get("total_losses", 0)
    wr          = metrics.get("win_rate", 0)
    avg_win     = metrics.get("avg_win", 0)
    avg_loss    = metrics.get("avg_loss", 0)
    rr          = metrics.get("rr_ratio", 0)
    avg_hold    = metrics.get("avg_hold_h", 0)
    stop_hits   = metrics.get("stop_hits", 0)

    print(f"  Win Rate      : {wr:.1f}%  ({total_wins}W / {total_losses}L  out of {total_rt} round-trips)")
    print(f"  Avg Win       : ${avg_win:+.2f}")
    print(f"  Avg Loss      : ${avg_loss:+.2f}")
    print(f"  Reward/Risk   : {rr:.2f}x  {'✅' if rr >= 1.0 else '⚠️  Need ≥1.0x to be profitable'}")
    print(f"  Avg Hold Time : {avg_hold:.1f} hrs")
    if stop_hits > 0:
        print(f"  🛑 Stop-Loss Hits : {stop_hits}  (stop-loss IS working)")

    if "total_value_traded" in metrics:
        print(f"  Total $ Traded : ${metrics['total_value_traded']:,.2f}")

    # ── Round-trip detail per symbol ──
    if "pnl_by_symbol" in metrics:
        for symbol, data in sorted(metrics["pnl_by_symbol"].items()):
            rt_df = data.get("rt_df")
            if rt_df is None or rt_df.empty:
                continue
            print(f"\n  📋 {symbol} Round-Trip Detail:")
            print(f"  {'Buy':>16}  {'Sell':>16}  {'Qty':>4}  {'Buy$':>7}  {'Sell$':>7}  {'P&L':>8}  {'Hold':>6}  {'Reason'}")
            print(f"  {'-'*85}")
            for _, t in rt_df.iterrows():
                flag = "✅" if t["win"] else "❌"
                bt   = str(t["buy_time"])[:16]
                st   = str(t["sell_time"])[:16]
                rsn  = t["exit_reason"]
                print(f"  {bt:>16}  {st:>16}  {t['qty']:>4.0f}  {t['buy_price']:>7.2f}  {t['sell_price']:>7.2f}  {t['pnl']:>+8.2f}  {t['hold_hours']:>5.1f}h  {rsn} {flag}")

    print("\n" + "=" * 70)
    print("💡 INSIGHTS:")

    # Insights
    if metrics.get("reconciliations", 0) > 0:
        print(f"  ⚠️  {metrics['reconciliations']} reconciliation(s) detected")
    else:
        print("  ✅ No reconciliations needed")

    if metrics.get("contradictions_logged", 0) > 0:
        print(f"  ⚠️  {metrics['contradictions_logged']} contradiction(s) logged")

    wr = metrics.get("win_rate", 0)
    rr = metrics.get("rr_ratio", 0)
    if wr >= 60:
        print(f"  ✅ Strong win rate: {wr:.1f}%")
    elif wr >= 50:
        print(f"  👍 Decent win rate: {wr:.1f}%")
    elif wr > 0:
        print(f"  ⚠️  Low win rate: {wr:.1f}% — review signal thresholds")

    if rr > 0 and rr < 1.0:
        print(f"  ⚠️  R/R ratio {rr:.2f}x is below 1.0 — losses outsize wins")
        print(f"      → Consider tighter stop-loss or wider take-profit")

    avg_hold = metrics.get("avg_hold_h", 0)
    if avg_hold < 0.5:
        print(f"  ⚠️  Avg hold {avg_hold:.1f}h is very short — possible noise trading")
        print(f"      → Consider MIN_HOLD_MINUTES = 15 guard in execute_sell()")

    stop_hits = metrics.get("stop_hits", 0)
    if stop_hits > 0:
        pct = stop_hits / metrics.get("total_rt", 1) * 100
        print(f"  🛑 Stop-loss fired {stop_hits}x ({pct:.0f}% of trades) — working as intended")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze trading performance")
    parser.add_argument("--days",   type=int, help="Analyze last N days")
    parser.add_argument("--symbol", type=str, help="Analyze specific symbol")
    parser.add_argument("--export", action="store_true", help="Export to CSV")
    args = parser.parse_args()

    print("\n🔍 Loading trade logs...")
    if args.symbol:
        print(f"   Symbol: {args.symbol.upper()}")
    if args.days:
        print(f"   Period: Last {args.days} days")

    df = load_trade_logs(symbol=args.symbol, days=args.days)

    if df is None or df.empty:
        print("\n❌ No trade data found\n")
        return

    print(f"   ✅ Loaded {len(df)} records")

    metrics = calculate_metrics(df)
    print_report(metrics, days=args.days)

    if args.export:
        output_file = f"trade_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✅ Exported to: {output_file}\n")


if __name__ == "__main__":
    main()
