import tensorflow as tf
from config import HISTORY_SIZE, FEATURES
from tensorflow.keras.layers import BatchNormalization


def get_lstm():
    """
    Keras LSTM Architecture
    """
    shape = (HISTORY_SIZE, len(FEATURES))
    ssm = tf.keras.models.Sequential()
    ssm.add(tf.keras.layers.LSTM(32, return_sequences=True,
                                 input_shape=shape))
    ssm.add(BatchNormalization())
    ssm.add(tf.keras.layers.LSTM(16))
    ssm.add(BatchNormalization())
    ssm.add(tf.keras.layers.Dense(32))
    ssm.add(tf.keras.layers.Dense(4, activation='softmax'))

    ssm.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
    return ssm
