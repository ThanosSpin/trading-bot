import os
import json
import csv
from config import SYMBOL, PORTFOLIO_PATH

OLD_PORTFOLIO_FILE = PORTFOLIO_PATH   # e.g., data/portfolio.json
OLD_TRADE_LOG = "data/trade_log.csv"

def migrate_portfolio(symbol):
    folder = os.path.dirname(PORTFOLIO_PATH)

    # New file paths
    new_portfolio_file = os.path.join(folder, f"portfolio_{symbol}.json")
    new_trade_log = os.path.join(folder, f"trades_{symbol}.csv")

    # ---- Portfolio ----
    if os.path.exists(OLD_PORTFOLIO_FILE):
        with open(OLD_PORTFOLIO_FILE, "r") as f:
            portfolio = json.load(f)
        with open(new_portfolio_file, "w") as f:
            json.dump(portfolio, f, indent=4)
        print(f"✅ Migrated portfolio → {new_portfolio_file}")
    else:
        print("⚠️ No old portfolio.json found to migrate.")

    # ---- Trade Log ----
    if os.path.exists(OLD_TRADE_LOG):
        with open(OLD_TRADE_LOG, "r") as infile, open(new_trade_log, "w", newline="") as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            for i, row in enumerate(reader):
                if i == 0:
                    # Update header to include symbol column if missing
                    if "symbol" not in row:
                        row.insert(1, "symbol")
                    writer.writerow(row)
                else:
                    if len(row) < 5:  # legacy format without symbol column
                        timestamp, action, price, value = row
                        writer.writerow([timestamp, symbol, action, price, value])
                    else:
                        writer.writerow(row)

        print(f"✅ Migrated trades → {new_trade_log}")
    else:
        print("⚠️ No old trade_log.csv found to migrate.")


if __name__ == "__main__":
    symbols = SYMBOL if isinstance(SYMBOL, list) else [SYMBOL]
    for sym in symbols:
        migrate_portfolio(sym)