import tensorflow as tf
# import pyfinancialdata
import numpy as np

from dataset import get_datasets, EquityData, split_multivariate
from callbacks import TENSORBOARD_CB, VALIDATION_CB, GetWeights
from model import get_lstm
from config import BATCH_SIZE, BUFFER_SIZE, \
    EPOCHS, HISTORY_SIZE, TARGET_DIS, \
    STEP, FEATURES, DATA_DIR, \
    DATA_SYM

tf.random.set_seed(42)

if __name__ == "__main__":
    
    # set up
    e = EquityData(DATA_DIR, DATA_SYM)
    # e.data['MA_long'] = e.data['Close'].rolling(window=52).mean()
    e.data['MA_short'] = e.data['Close'].rolling(window=7).mean()
    # evaluation interval
    window = int(e.data.shape[0] / BATCH_SIZE)
    t_ds, v_ds = get_datasets(e) 

    ssm = get_lstm()

    # run trial
    history = ssm.fit(t_ds, epochs=EPOCHS,
                      steps_per_epoch=window,
                      validation_data=v_ds,
                      validation_steps= 20,
                      callbacks=[VALIDATION_CB, TENSORBOARD_CB])

