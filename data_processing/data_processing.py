import numpy as np
import tensorflow as tf
from config import BATCH_SIZE, FEATURES, \
                   HISTORY_SIZE, TARGET_DIS, STEP, BUFFER_SIZE


def create_time_steps(length):
    return list(range(-length, 0))


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step):
    data = []
    labels = []
    std_close = dataset[:, 0].std() / 10.0

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if target[i + target_size] > target[i] + std_close:
            labels.append([1, 0, 0, 0])
        elif target[i + target_size] >= target[i] and target[i + target_size] < (target[i] + std_close):  # went up
            # positive, less than 1/10th std
            labels.append([0, 1, 0, 0])
        elif target[i + target_size] < target[i] - std_close:
            # greater than 1/10 std negative change
            labels.append([0, 0, 1, 0])
        elif target[i + target_size] < target[i] and target[i + target_size] > (target[i] - std_close):
            labels.append([0, 0, 0, 1])
        else:
            print(target[i], target[i + target_size], std_close)
            print("Error labelling")
            exit(1)

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


def add_features(e):
    e.data['MA_short'] = e.data['Close'].rolling(window=7).mean()
    e.data['Change_1'] = e.data['Close'] - e.data['Close'].shift(1)
    e.data['Change_4'] = e.data['Close'] - e.data['Close'].shift(4)
    e.data['Change_8'] = e.data['Close'] - e.data['Close'].shift(8)
    e.data = e.data[10:]
    return e


def get_datasets(e):
    dataset = e.data[FEATURES].values
    return split_multivariate(dataset,
                              HISTORY_SIZE,
                              TARGET_DIS,
                              STEP)


def get_tfds(x_train, y_train, x_val, y_val):
    t_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    t_ds = t_ds.cache().shuffle(
        BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    v_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    v_ds = v_ds.batch(BATCH_SIZE).repeat()

    return t_ds, v_ds
