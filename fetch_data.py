import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(tickers: list, start_date, interval = '1d') -> dict:
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start_date, interval=interval, multi_level_index=False)
        if df.empty:
            print(f"Empty DataFrame for {ticker}")
            continue
        else:
            data[ticker] = df 
        data[ticker]['returns'] = data[ticker]['Close'].pct_change()

    return data