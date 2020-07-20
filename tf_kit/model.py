import tensorflow as tf
from config import HISTORY_SIZE, FEATURES, N_CLASSES
from tensorflow.keras.layers import BatchNormalization, Dropout
import tensorflow_addons as tfa


def get_lstm():
    """
    Keras LSTM Architecture
    """
    shape = (HISTORY_SIZE, len(FEATURES))
    ssm = tf.keras.models.Sequential()
    ssm.add(tf.keras.layers.LSTM(32, return_sequences=True,
                                 input_shape=shape))

    ssm.add(tf.keras.layers.LSTM(32))

    ssm.add(tf.keras.layers.Dense(32))
    ssm.add(tf.keras.layers.Dense(16))
    ssm.add(tf.keras.layers.Dense(N_CLASSES, activation='softmax'))
    ssm.compile(optimizer=tf.keras.optimizers.Adam(lr=0.003),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy', tfa.metrics.F1Score(num_classes=N_CLASSES)])
    return ssm
