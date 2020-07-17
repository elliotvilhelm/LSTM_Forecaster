import yfinance as yf
import pandas as pd
from data_processing.data_processing import get_datasets
from market.Equity import EquityData
from data_processing.data_processing import add_features
from config import HISTORY_SIZE, FEATURES, N_CLASSES
import numpy as np
import time


def download_data(tickers, interval="1h"):
    for ticker in tickers:
        df = yf.download(ticker, start="2018-08-15", end=None, interval="1h")
        print(f"Downloading: {ticker}")
        df.to_csv(f"data/{ticker}_2018_2020_1hr.csv")
        time.sleep(1)


def get_ohlc(ticker, period, interval):
    t = yf.Ticker(ticker)
    print("history: {}\ninterval: {}".format(period, interval))
    history = t.history(period=period, interval=interval)
    history = history[["Open", "High", "Low", "Close", "Volume"]]
    return history


def get_multi_df(tickers):
    x_train_total = np.empty(shape=(0, HISTORY_SIZE, len(FEATURES)))
    y_train_total = np.empty(shape=(0, N_CLASSES))
    x_val_total = np.empty(shape=(0, HISTORY_SIZE, len(FEATURES)))
    y_val_total = np.empty(shape=(0, N_CLASSES))
    for ticker in tickers:
        e = EquityData(f"data/{ticker}_2018_2020_1hr.csv", ticker)
        e = add_features(e)
        x_train, y_train, x_val, y_val = get_datasets(e)
        x_train_total = np.append(x_train_total, x_train, axis=0)
        y_train_total = np.append(y_train_total, y_train, axis=0)
        x_val_total = np.append(x_val_total, x_val, axis=0)
        y_val_total = np.append(y_val_total, y_val, axis=0)

    return x_train_total, y_train_total, x_val_total, y_val_total



