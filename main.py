import tensorflow as tf
from data_collection.yfinance_collector import get_multi_df
from analysis.distribution_analysis import log_distributions
from data_processing.data_processing import get_tfds, down_sample_three_class_data
from config import BATCH_SIZE, EPOCHS, TICKERS
from tf_kit.callbacks import TENSORBOARD_CB, VALIDATION_CB, GetConfusion
from tf_kit.model import get_lstm

tf.random.set_seed(42)

x_train, y_train, x_val, y_val = get_multi_df(TICKERS)
log_distributions(x_train, y_train, x_val, y_val)
x_train, y_train = down_sample_three_class_data(x_train, y_train)
x_val, y_val = down_sample_three_class_data(x_val, y_val)
CONFUSION_CB = GetConfusion(x_val, y_val)
log_distributions(x_train, y_train, x_val, y_val)

tfds_train, tfds_val = get_tfds(x_train, y_train, x_val, y_val)
window = int(x_train.shape[0] / BATCH_SIZE)
ssm = get_lstm()
history = ssm.fit(tfds_train, epochs=EPOCHS,
                  steps_per_epoch=window,
                  validation_data=tfds_val,
                  validation_steps=int(y_val.shape[0]/BATCH_SIZE),
                  callbacks=[VALIDATION_CB, TENSORBOARD_CB, CONFUSION_CB])
