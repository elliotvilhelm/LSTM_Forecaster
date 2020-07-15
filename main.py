import tensorflow as tf
# import pyfinancialdata
from datetime import datetime as dt
import numpy as np

from tensorflow.keras.layers import BatchNormalization
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import register_matplotlib_converters

from config import BATCH_SIZE, BUFFER_SIZE, \
    EPOCHS, HISTORY_SIZE, TARGET_DIS, \
    STEP, FEATURES, DATA_DIR, \
    DATA_SYM

register_matplotlib_converters()
tf.random.set_seed(42)


class GetWeights(tf.keras.callbacks.Callback):
    # Keras callback which collects values of weights and biases at each epoch
    def __init__(self):
        super(GetWeights, self).__init__()
        self.weight_dict = {}

    def on_epoch_end(self, epoch, logs=None):
        # this function runs at the end of each epoch

        # loop over each layer and get weights and biases
        for layer_i in range(len(self.model.layers)):
            w = self.model.layers[layer_i].get_weights()[0]
            b = self.model.layers[layer_i].get_weights()[1]
            print('Layer %s has weights of shape %s and biases of shape %s' %(
                layer_i, np.shape(w), np.shape(b)))
            print(w)
            # save all weights and biases inside a dictionary
            if epoch == 0:
                # create array to hold weights and biases
                self.weight_dict['w_'+str(layer_i+1)] = w
                self.weight_dict['b_'+str(layer_i+1)] = b
            else:
                # append new weights to previously-created weights array
                self.weight_dict['w_'+str(layer_i+1)] = np.dstack(
                    (self.weight_dict['w_'+str(layer_i+1)], w))
                # append new weights to previously-created weights array
                self.weight_dict['b_'+str(layer_i+1)] = np.dstack(
                    (self.weight_dict['b_'+str(layer_i+1)], b)) 


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
        if target[i] <= target[i+target_size]: # went up
            labels.append(1)
        else:
            labels.append(0)

    return np.array(data), np.array(labels)


def split_multivariate(dataset, history_size, target_distance, step):
    train_split = int(len(dataset) * 0.7)

    # data_mean = dataset[:train_split].mean(axis=0)
    # data_std = dataset[:train_split].std(axis=0)
    # dataset = (dataset - data_mean) / data_std

    x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 0], 0,
                                                       train_split, history_size,
                                                       target_distance, step)
    x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 0],
                                                   train_split, None, history_size,
                                                   target_distance, step)

    return x_train_single, y_train_single, x_val_single, y_val_single


def create_time_steps(length):
    return list(range(-length, 0))


def get_lstm():
    """
    Keras LSTM Architecture
    """
    shape = (HISTORY_SIZE, len(FEATURES))
    ssm = tf.keras.models.Sequential()
    ssm.add(tf.keras.layers.LSTM(32, return_sequences=True,
                                 input_shape=shape))
    ssm.add(BatchNormalization())

    #ssm.add(tf.keras.layers.LSTM(32, return_sequences=True))

    ssm.add(tf.keras.layers.LSTM(16))
    ssm.add(BatchNormalization())

    ssm.add(tf.keras.layers.Dense(8))
    ssm.add(BatchNormalization())

    ssm.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    ssm.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=['accuracy'])
    return ssm


if __name__ == "__main__":
    e = EquityData(DATA_DIR, DATA_SYM)
    # e.data['MA_long'] = e.data['Close'].rolling(window=52).mean()
    e.data['MA_short'] = e.data['Close'].rolling(window=7).mean()

    # evaluation interval
    window = int(e.data.shape[0] / BATCH_SIZE) * 1

    # pick selected features
    e.data = e.data[8:]
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

    # validation callback
    validation_cb = tf.keras.callbacks.ModelCheckpoint(
        'checkpoints/multivariate_single_model', monitor='val_accuracy', verbose=1, save_best_only=True,
        save_weights_only=False, mode='auto', save_freq='epoch'
    )

    # tensorboard callback
    logdir = "logs/scalars/{}".format(dt.today())
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    gb = GetWeights()
    # get model
    ssm = get_lstm()

    # run trial
    history = ssm.fit(t_ds, epochs=EPOCHS,
                      steps_per_epoch=window,
                      validation_data=v_ds,
                      validation_steps= 20,
                      callbacks=[validation_cb, tensorboard_cb, gb])
    
    state = v_ds.take(1)
    for x, y in state:
        print(x, y)
        print(ssm.predict(x))
