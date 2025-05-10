import yfinance as yf
import pandas as pd

def get_stock_data(symbol, days=30):
    df = yf.download(symbol, period=f"{days}d", interval="1d")
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    return df
