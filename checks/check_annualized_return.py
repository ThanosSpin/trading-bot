import pandas as pd
from portfolio import get_daily_portfolio_file
import os

# Daily portfolio
port_path = get_daily_portfolio_file()
df = pd.read_csv(port_path)
df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
df = df.sort_values("date").reset_index(drop=True)

valuecol = next(c for c in ["value", "equity", "totalvalue", "portfoliovalue"] if c in df.columns)
df["total_equity"] = pd.to_numeric(df[valuecol], errors="coerce").fillna(0.0)

# Deposits (same logic as dashboard)
deposit_path = os.path.join(os.path.dirname(port_path), "deposits.csv")
if os.path.exists(deposit_path):
    df_dep = pd.read_csv(deposit_path)
    df_dep["date"] = pd.to_datetime(df_dep["date"], utc=True, errors="coerce")
    df_dep = df_dep.sort_values("date").reset_index(drop=True)
else:
    df_dep = pd.DataFrame(columns=["date", "amount"])

# Start with zero externalflow per portfolio row
df["externalflow"] = 0.0

if not df_dep.empty:
    # For each deposit/withdrawal, add its amount to the nearest portfolio date >= deposit date
    for _, row in df_dep.iterrows():
        d = row["date"]
        amt = float(row["amount"])
        # Find the first portfolio row on or after this date
        idx = df.index[df["date"] >= d]
        if len(idx) > 0:
            df.loc[idx[0], "externalflow"] += amt

# Initial cash (same as dashboard)
if "initialcash" in df.columns:
    initialcash = pd.to_numeric(df["initialcash"], errors="coerce").fillna(0.0).iloc[0]
else:
    initialcash = float(df["total_equity"].iloc[0])

df["cumexternalflow"] = df["externalflow"].cumsum()
df["totaldeposited"] = initialcash + df["cumexternalflow"]

df["pnlvalue"] = df["total_equity"] - df["totaldeposited"]

print(df[["date", "total_equity", "cumexternalflow", "totaldeposited", "pnlvalue"]].tail(30))