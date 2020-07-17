import tensorflow as tf
from config import HISTORY_SIZE, FEATURES
from tensorflow.keras.layers import BatchNormalization, Dropout

def get_lstm():
    """
    Keras LSTM Architecture
    """
    shape = (HISTORY_SIZE, len(FEATURES))
    ssm = tf.keras.models.Sequential()
    ssm.add(tf.keras.layers.LSTM(128, return_sequences=True,
                                 input_shape=shape))

    ssm.add(tf.keras.layers.LSTM(64))

    ssm.add(tf.keras.layers.Dense(32))
    ssm.add(tf.keras.layers.Dense(16))
    ssm.add(tf.keras.layers.Dense(3, activation='softmax'))
    ssm.compile(optimizer=tf.keras.optimizers.Adam(lr=0.005),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
    return ssm
