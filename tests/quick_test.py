import pandas as pd
from datetime import datetime, timezone, timedelta

df = pd.read_csv("logs/predictions_NVDA.csv")
# df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
# print("Now UTC:", datetime.now(timezone.utc))
# print("Min ts:", df["timestamp"].min(), "Max ts:", df["timestamp"].max())

# print("Columns:", df.columns.tolist())

# # If actual_outcome doesn't exist, that's the core issue
# if "actual_outcome" not in df.columns:
#     print("actual_outcome column is MISSING")
# else:
#     print("actual_outcome NaN count (all rows):", df["actual_outcome"].isna().sum())

#     # Simulate the 168 hour lookback filter outcome_tracker uses
#     df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
#     from datetime import datetime, timezone, timedelta
#     cutoff = datetime.now(timezone.utc) - timedelta(hours=168)
#     df_update = df[df["timestamp"] > cutoff].copy()
#     print("Rows after 168h filter:", len(df_update))
#     print("NaNs in actual_outcome after filter:", df_update["actual_outcome"].isna().sum())



# import pandas as pd
# df = pd.read_csv("logs/predictions_ABBV_old.csv")
print(df.columns.tolist())

# Show how many rows got labeled
print("NVDA non-NaN actual_outcome:", df["actual_outcome"].notna().sum())

# # See a few labeled rows (only rows with outcome 0 or 1)
# print(df[df["actual_outcome"].notna()][
#     ["timestamp", "symbol", "mode", "predicted_prob", "price", "actual_outcome", "return_pct"]
# ].head(20))

daily = df[(df["symbol"] == "NVDA") & (df["mode"] == "daily")]
print(daily["actual_outcome"].value_counts(normalize=True))