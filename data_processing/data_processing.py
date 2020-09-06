import numpy as np
import tensorflow as tf
from ta import add_all_ta_features
from ta.utils import dropna
import numpy


from trading.strategy import ThreeClassPrediction
from analysis.distribution_analysis import log_distributions
from config import BATCH_SIZE, FEATURES, \
                   HISTORY_SIZE, TARGET_DIS, \
                   STEP, BUFFER_SIZE, STD_RATIO
import pandas as pd


def create_time_steps(length):
    return list(range(-length, 0))


def down_sample_three_class_data(x, y):
    y_df = pd.DataFrame(y.argmax(axis=1), columns=['label'])
    y_min_count = y_df['label'].value_counts().min()
    up_indicies = y_df[y_df['label'] == 0].index.values
    chop_indicies = y_df[y_df['label'] == 1].index.values
    down_indicies = y_df[y_df['label'] == 2].index.values
    up_indicies = np.random.choice(up_indicies, y_min_count)
    chop_indicies = np.random.choice(chop_indicies, y_min_count)
    down_indicies = np.random.choice(down_indicies, y_min_count)
    up_x = np.take(x, up_indicies, axis=0)
    chop_x = np.take(x, chop_indicies, axis=0)
    down_x = np.take(x, down_indicies, axis=0)
    up_y = np.take(y, up_indicies, axis=0)
    chop_y = np.take(y, chop_indicies, axis=0)
    down_y = np.take(y, down_indicies, axis=0)
    x = np.concatenate((up_x, chop_x, down_x))
    y = np.concatenate((up_y, chop_y, down_y))
    return x, y


def preprocess(ds, std_close, start, end):
    data = []
    start = start + HISTORY_SIZE
    if end is None:
        end = len(ds) - TARGET_DIS

    for i in range(start, end):
        indices = range(i-HISTORY_SIZE, i, STEP)
        sequence = ds[indices]
        normalized_sequence = (sequence / sequence[0]) - 1
        data.append(normalized_sequence)
        if numpy.isnan(normalized_sequence).any():
            print(sequence)
            print(normalized_sequence)
            exit(1)

    s = ThreeClassPrediction(ds, std_close)
    labels = s.add_labels(start, end)
    
    return np.array(data), np.array(labels)


def train_test_split(ds):
    train_split = int(len(ds) * 0.8)

    std_close = ds[:train_split].std(axis=0)[0] / STD_RATIO
    x_t, y_t = preprocess(ds, std_close, 0, train_split)
    x_v, y_v = preprocess(ds, std_close, train_split, None)
    return x_t, y_t, x_v, y_v


def add_features(e):
    e.data = dropna(e.data)
    e.data = add_all_ta_features(e.data, 
                                 open="Open", 
                                 high="High", 
                                 low="Low", 
                                 close="Close",
                                 volume="Volume")

    # 43 for trend_trix
    e.data = e.data[43:]
    return e


def get_datasets(e):
    ds = e.data[FEATURES].values
    return train_test_split(ds)


def get_tfds(x_train, y_train, x_val, y_val):
    t_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    v_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    t_ds = t_ds.shuffle(BUFFER_SIZE).cache().batch(BATCH_SIZE).repeat()
    v_ds = v_ds.batch(BATCH_SIZE).repeat()

    return t_ds, v_ds
