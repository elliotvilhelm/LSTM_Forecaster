import pandas as pd
import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
from pandas.plotting import register_matplotlib_converters
from config import DATA_DIR, DATA_SYM, BATCH_SIZE, FEATURES, \
                   HISTORY_SIZE, TARGET_DIS, STEP, BUFFER_SIZE
from datetime import datetime as dt

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])
        if target[i] <= target[i+target_size]:  # went up
            labels.append(1)
        else:
            labels.append(0)

    return np.array(data), np.array(labels)

def split_multivariate(dataset, history_size, target_distance, step):
    train_split = int(len(dataset) * 0.7)

    data_mean = dataset[:train_split].mean(axis=0)
    data_std = dataset[:train_split].std(axis=0)
    dataset = (dataset - data_mean) / data_std
    dataset = (dataset - dataset.min(axis=0)) / (dataset.max(axis=0) - dataset.min(axis=0))

    x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 0], 0,
                                                       train_split, history_size,
                                                       target_distance, step)
    x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 0],
                                                   train_split, None, history_size,
                                                   target_distance, step)

    return x_train_single, y_train_single, x_val_single, y_val_single

class EquityData:
    def __init__(self, filename, symbol):
        """
        filename: .csv file of OHLC values
        symbol: ticker
        """
        self.symbol = symbol
        self.data = self.load_csv(filename)

    def load_csv(self, filename):
        """
        filename: .csv file of OHLC values
        """
        return pd.read_csv(filename)

    def close(self):
        return self.data['Close']

    def date(self):
        return self.data['Date']

    def plot(self, attr='Close'):
        df = pd.DataFrame()
        date_time = pd.to_datetime(self.date())
        df['Close'] = self.close()
        df = df.set_index(date_time)
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # to get a tick every 15 minutes
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_tick_params(rotation=45)
        fig.subplots_adjust(bottom=0.3)
        plt.plot(df)
        plt.show()

def get_datasets():
    e = EquityData(DATA_DIR, DATA_SYM)
    # e.data['MA_long'] = e.data['Close'].rolling(window=52).mean()
    e.data['MA_short'] = e.data['Close'].rolling(window=7).mean()

    # evaluation interval
    window = int(e.data.shape[0] / BATCH_SIZE)

    # pick selected features
    e.data['Change_1'] = e.data['Close'] - e.data['Close'].shift(1)
    e.data['Change_4'] = e.data['Close'] - e.data['Close'].shift(4)
    e.data['Change_8'] = e.data['Close'] - e.data['Close'].shift(8)
    e.data = e.data[10:]

    features = e.data[FEATURES]
    dataset = features.values

    # get validation and training data
    xt, yt, xv, yv = split_multivariate(dataset,
                                        HISTORY_SIZE,
                                        TARGET_DIS,
                                        STEP)

    # construct datasets
    t_ds = tf.data.Dataset.from_tensor_slices((xt, yt))
    t_ds = t_ds.cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    v_ds = tf.data.Dataset.from_tensor_slices((xv, yv))
    v_ds = v_ds.batch(BATCH_SIZE).repeat()
    
    return t_ds, v_ds, window