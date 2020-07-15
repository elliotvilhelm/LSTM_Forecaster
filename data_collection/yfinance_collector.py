import yfinance as yf
import pandas as pd


def download_data():
    tickers = ["SPY", "ROKU", "NVDA", "NFLX", "MSFT", "QQQ"]

    for ticker in tickers:
        df = yf.download(ticker, start="2018-08-15", end=None, interval="1h")
        print(f"Downloading: {ticker}")
        df.to_csv(f"../data/{ticker}_2018_2020_1hr.csv")


def get_ohlc(ticker, period, interval):
    t = yf.Ticker(ticker)
    print(period)
    print(interval)
    history = t.history(period=period, interval=interval)
    history = history[["Open", "High", "Low", "Close", "Volume"]]
    return history
