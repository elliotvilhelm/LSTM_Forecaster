import yfinance as yf
import pandas as pd

years = [2020, 2019]
tickers = ["SPY", "ROKU", "NVDA", "NFLX", "MSFT", "QQQ"]

for ticker in tickers:
    df = yf.download(ticker, start="2018-08-15", end=None, interval="1h")
    print(f"Downloading: {ticker}")
    df.to_csv(f"../data/{ticker}_2018_2020_1hr.csv")
