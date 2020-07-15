import tensorflow as tf
from datetime import datetime as dt

# validation callback
VALIDATION_CB = tf.keras.callbacks.ModelCheckpoint(
    'checkpoints/multivariate_single_model', monitor='val_accuracy', verbose=1, save_best_only=True,
    save_weights_only=False, mode='auto', save_freq='epoch'
)

# tensorboard callback
logdir = "logs/scalars/{}".format(dt.today())
TENSORBOARD_CB = tf.keras.callbacks.TensorBoard(log_dir=logdir)
