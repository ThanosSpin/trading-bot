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

def load_trade_logs(symbol=None, days=None):
    """Load trade logs from CSV files."""
    # Search in multiple locations
    search_paths = [
        ".",                    # Current directory
        "data",                 # data folder
        "logs",                 # logs folder
        "../logs",              # Parent's logs folder
        "./logs/trades",        # logs/trades subfolder
    ]

    files = []

    for path in search_paths:
        if not os.path.exists(path):
            continue

        if symbol:
            pattern = os.path.join(path, f"trades_{symbol.upper()}.csv")
            files.extend(glob.glob(pattern))
        else:
            pattern = os.path.join(path, "trades_*.csv")
            files.extend(glob.glob(pattern))

    # Remove duplicates
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

            # Handle different timestamp column names
            timestamp_col = None
            for col in ['timestamp', 'date', 'time', 'datetime']:
                if col in df.columns:
                    timestamp_col = col
                    break

            if timestamp_col:
                # ✅ FIX: Use format='ISO8601' for flexible parsing
                try:
                    df['timestamp'] = pd.to_datetime(df[timestamp_col], format='ISO8601')
                except:
                    # Fallback to infer format
                    try:
                        df['timestamp'] = pd.to_datetime(df[timestamp_col], format='mixed')
                    except:
                        # Last resort - let pandas infer
                        df['timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')

            # Extract symbol from filename if not in data
            if 'symbol' not in df.columns:
                sym = os.path.basename(file).replace('trades_', '').replace('.csv', '')
                df['symbol'] = sym.upper()

            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {file}: {e}")

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)

    # Filter by days if specified
    if days and 'timestamp' in df.columns:
        cutoff = datetime.now() - timedelta(days=days)
        df = df[df['timestamp'] >= cutoff]

    return df

def calculate_metrics(df):
    """Calculate trading performance metrics."""
    if df is None or df.empty:
        return None

    metrics = {}

    # Filter to actual trades (not reconciliations)
    if 'action' in df.columns:
        trades = df[df['action'].str.upper().isin(['BUY', 'SELL'])].copy()
    else:
        trades = df.copy()

    metrics['total_trades'] = len(trades)

    if metrics['total_trades'] == 0:
        print("⚠️ No trades found in the specified period")
        return metrics

    # Calculate P&L if price and qty available
    price_col = None
    qty_col = None

    # Find price column (could be 'price', 'Price', 'fill_price', etc.)
    for col in df.columns:
        if 'price' in col.lower():
            price_col = col
            break

    # Find qty column
    for col in df.columns:
        if col.lower() in ['qty', 'quantity', 'shares', 'amount']:
            qty_col = col
            break

    if price_col and qty_col:
        trades[price_col] = pd.to_numeric(trades[price_col], errors='coerce')
        trades[qty_col] = pd.to_numeric(trades[qty_col], errors='coerce')

        # Remove rows with invalid data
        trades = trades.dropna(subset=[price_col, qty_col])

        # Convert action to numeric (BUY = -1, SELL = +1)
        trades['direction'] = trades['action'].str.upper().map({'BUY': -1, 'SELL': 1})
        trades['value'] = trades[price_col].astype(float) * trades[qty_col].astype(float) * trades['direction']

        metrics['total_value_traded'] = abs(trades['value']).sum()

        # Group by symbol to calculate P&L
        pnl_by_symbol = {}

        for symbol in trades['symbol'].unique():
            sym_trades = trades[trades['symbol'] == symbol].copy()
            if 'timestamp' in sym_trades.columns:
                sym_trades = sym_trades.sort_values('timestamp')

            position = 0
            avg_price = 0
            realized_pnl = 0

            for _, trade in sym_trades.iterrows():
                qty = float(trade[qty_col])
                price = float(trade[price_col])
                action = str(trade['action']).upper()

                if action == 'BUY':
                    # Update average price
                    if position > 0:
                        avg_price = (avg_price * position + price * qty) / (position + qty)
                    else:
                        avg_price = price
                    position += qty

                elif action == 'SELL':
                    if position > 0:
                        # Realize profit/loss
                        pnl = (price - avg_price) * qty
                        realized_pnl += pnl
                        position -= qty

            pnl_by_symbol[symbol] = {
                'realized_pnl': realized_pnl,
                'position': position,
                'avg_price': avg_price
            }

        metrics['pnl_by_symbol'] = pnl_by_symbol
        metrics['total_realized_pnl'] = sum(s['realized_pnl'] for s in pnl_by_symbol.values())

        # Win rate
        winning_trades = sum(1 for s in pnl_by_symbol.values() if s['realized_pnl'] > 0)
        metrics['win_rate'] = (winning_trades / len(pnl_by_symbol)) * 100 if pnl_by_symbol else 0

    # Count by action
    if 'action' in trades.columns:
        action_counts = trades['action'].str.upper().value_counts()
        metrics['buys'] = action_counts.get('BUY', 0)
        metrics['sells'] = action_counts.get('SELL', 0)

    # Count reconciliations if present
    if 'action' in df.columns:
        reconcile_count = len(df[df['action'].str.upper() == 'RECONCILE'])
        metrics['reconciliations'] = reconcile_count

    # Count contradictions if logged
    if 'notes' in df.columns:
        contradiction_count = df['notes'].astype(str).str.contains('contradiction', case=False, na=False).sum()
        metrics['contradictions_logged'] = contradiction_count

    return metrics

def print_report(metrics, days=None):
    """Print formatted performance report."""
    print("\n" + "="*70)
    print(f"📊 TRADING PERFORMANCE REPORT")
    if days:
        print(f"Period: Last {days} days")
    print("="*70)

    if not metrics or metrics.get('total_trades', 0) == 0:
        print("\n⚠️ No trades to analyze")
        print("="*70 + "\n")
        return

    print(f"\n📈 TRADE STATISTICS:")
    print(f"  Total Trades: {metrics.get('total_trades', 0)}")
    print(f"  Buys:  {metrics.get('buys', 0)}")
    print(f"  Sells: {metrics.get('sells', 0)}")

    if 'reconciliations' in metrics and metrics['reconciliations'] > 0:
        print(f"  Reconciliations: {metrics['reconciliations']}")

    if 'contradictions_logged' in metrics and metrics['contradictions_logged'] > 0:
        print(f"  Contradictions: {metrics['contradictions_logged']}")

    print(f"\n💰 PROFIT & LOSS:")

    if 'total_realized_pnl' in metrics:
        pnl = metrics['total_realized_pnl']
        print(f"  Total Realized P&L: ${pnl:,.2f}")

        if 'pnl_by_symbol' in metrics:
            print(f"\n  By Symbol:")
            for symbol, data in sorted(metrics['pnl_by_symbol'].items()):
                pnl_sym = data['realized_pnl']
                pos = data['position']
                avg = data['avg_price']

                pnl_str = f"${pnl_sym:+,.2f}" if pnl_sym != 0 else "$0.00"
                print(f"    {symbol:6s}: {pnl_str:>12s}  (Position: {pos:>5g} @ ${avg:.2f})")

        if 'win_rate' in metrics:
            print(f"\n  Win Rate: {metrics['win_rate']:.1f}%")
    else:
        print(f"  ⚠️ Could not calculate P&L (missing price/qty columns)")

    if 'total_value_traded' in metrics:
        print(f"\n  Total Value Traded: ${metrics['total_value_traded']:,.2f}")

    print("\n" + "="*70)
    print("💡 INSIGHTS:")

    # Provide insights
    if metrics.get('reconciliations', 0) > 0:
        print(f"  ⚠️ {metrics['reconciliations']} reconciliation(s) detected")
        print(f"     (Check for partial fills or manual trades)")
    else:
        print(f"  ✅ No reconciliations needed (portfolios stayed in sync)")

    if metrics.get('contradictions_logged', 0) > 0:
        print(f"  ⚠️ {metrics['contradictions_logged']} contradiction(s) logged")
        print(f"     (Review if contradictions are frequent >20%)")

    if 'win_rate' in metrics and metrics['win_rate'] > 0:
        wr = metrics['win_rate']
        if wr >= 60:
            print(f"  ✅ Strong win rate: {wr:.1f}%")
        elif wr >= 50:
            print(f"  👍 Decent win rate: {wr:.1f}%")
        else:
            print(f"  ⚠️ Low win rate: {wr:.1f}% (review strategy)")

    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze trading performance")
    parser.add_argument('--days', type=int, help='Analyze last N days')
    parser.add_argument('--symbol', type=str, help='Analyze specific symbol')
    parser.add_argument('--export', action='store_true', help='Export to CSV')
    args = parser.parse_args()

    print(f"\n🔍 Loading trade logs...")
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