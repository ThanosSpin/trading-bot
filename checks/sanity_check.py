#!/usr/bin/env python3

import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# Load daily portfolio
# -------------------------------------------------------------------
df = pd.read_csv("data/daily_portfolio.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Ensure date is timezone-aware DatetimeIndex (align with dashboard)
df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
df = df.dropna(subset=["date"]).reset_index(drop=True)

# Expect columns: date, value, cash, external_flow, initial_cash
# -------------------------------------------------------------------
# Rebuild net_cash_flow and pnl_value (pure trading PnL)
# -------------------------------------------------------------------
df["external_flow"] = pd.to_numeric(df.get("external_flow", 0.0), errors="coerce").fillna(0.0)
df["value"] = pd.to_numeric(df.get("value", 0.0), errors="coerce").fillna(0.0)

# Initial cash from file (can be 0 if you rely only on deposits)
initial_cash = float(pd.to_numeric(df.get("initial_cash", 0.0).iloc[0], errors="coerce") or 0.0)

df["cum_external"] = df["external_flow"].cumsum()
df["net_cash_flow"] = initial_cash + df["cum_external"]

# PnL-only equity (equity with deposits stripped)
df["pnl_value"] = df["value"] - df["net_cash_flow"]

# -------------------------------------------------------------------
# Window and returns
# -------------------------------------------------------------------
start_date = df["date"].iloc[0]
end_date = df["date"].iloc[-1]
start_equity = float(df["value"].iloc[0])
end_equity = float(df["value"].iloc[-1])
total_pnl = float(df["pnl_value"].iloc[-1])

# Base capital: implied from start equity and total PnL
# (same logic as dashboard fallback when deposits not used directly)
base_capital = max(start_equity - total_pnl, 1e-9)
total_return = total_pnl / base_capital

elapsed_days = max((end_date - start_date).total_seconds() / 86400.0, 1.0)
annual_return = (1.0 + total_return) ** (365.25 / elapsed_days) - 1.0 if total_return > -1 else -1.0

print(f"Start date:      {start_date}")
print(f"End date:        {end_date}")
print(f"Base capital:    {base_capital:,.2f}")
print(f"Total PnL:       {total_pnl:,.2f}")
print(f"Total return:    {100 * total_return:,.2f}%")
print(f"Annualized ret.: {100 * annual_return:,.2f}%")

# -------------------------------------------------------------------
# Strategy equity curve (pure PnL) and daily returns
# -------------------------------------------------------------------
df["strategy_equity"] = base_capital + df["pnl_value"]
df = df[df["strategy_equity"] > 0].copy()

daily = (
    df.set_index("date")[["strategy_equity"]]
    .resample("1D")
    .last()
    .dropna()
)

daily["ret"] = daily["strategy_equity"].pct_change()
rets = daily["ret"].replace([np.inf, -np.inf], np.nan).dropna()

if len(rets) >= 2 and rets.std() > 0:
    sharpe = (rets.mean() / rets.std()) * (252 ** 0.5)
    vol = rets.std() * (252 ** 0.5)
else:
    sharpe = None
    vol = None

print(f"Sharpe:          {sharpe:.4f}" if sharpe is not None else "Sharpe:          N/A")
print(f"Volatility:      {100 * vol:,.2f}%" if vol is not None else "Volatility:      N/A")

# -------------------------------------------------------------------
# Daily return distribution summary (pure PnL equity)
# -------------------------------------------------------------------
if len(rets) > 0:
    desc = rets.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    print("\nDaily return distribution (pure PnL equity):")
    print(f"  Days:       {int(desc['count'])}")
    print(f"  Min:        {desc['min']*100:6.2f}%")
    print(f"  5th pct:    {desc['5%']*100:6.2f}%")
    print(f"  25th pct:   {desc['25%']*100:6.2f}%")
    print(f"  Median:     {desc['50%']*100:6.2f}%")
    print(f"  75th pct:   {desc['75%']*100:6.2f}%")
    print(f"  95th pct:   {desc['95%']*100:6.2f}%")
    print(f"  Max:        {desc['max']*100:6.2f}%")
else:
    print("No valid daily returns to summarize.")
