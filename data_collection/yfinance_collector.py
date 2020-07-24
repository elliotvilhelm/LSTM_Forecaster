import yfinance as yf
from data_processing.data_processing import get_datasets
from market.Equity import EquityData
from data_processing.data_processing import add_features
from config import HISTORY_SIZE, FEATURES, N_CLASSES
import numpy as np
import time
import os


def download_data(tickers, interval="1h"):
    
    for ticker in tickers:
        file_path = f"data/{ticker}_2018_2020_1hr.csv"
        if not os.path.exists(file_path):
            print(f"Downloading: {ticker}")
            df = yf.download(ticker, start="2018-08-15", end=None, interval="1h")
            df.to_csv(file_path)
            time.sleep(1)
        else:
            print(f"{ticker} already downloaded.")


def get_ohlc(ticker, period, interval, start=None):
    t = yf.Ticker(ticker)
    print("history: {}\ninterval: {}".format(period, interval))
    history = t.history(period=period, interval=interval, start=start)
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



