import tensorflow as tf
# import pyfinancialdata
import numpy as np

from dataset import get_datasets, EquityData, split_multivariate
from callbacks import TENSORBOARD_CB, VALIDATION_CB
from model import get_lstm
from utils import GetWeights
from config import BATCH_SIZE, BUFFER_SIZE, \
    EPOCHS, HISTORY_SIZE, TARGET_DIS, \
    STEP, FEATURES, DATA_DIR, \
    DATA_SYM

tf.random.set_seed(42)

if __name__ == "__main__":
    
    # set up
    t_ds, v_ds, window = get_datasets() 
    ssm = get_lstm()

    # run trial
    history = ssm.fit(t_ds, epochs=EPOCHS,
                      steps_per_epoch=window,
                      validation_data=v_ds,
                      validation_steps= 20,
                      callbacks=[VALIDATION_CB, TENSORBOARD_CB])
    
    # validate
    state = v_ds.take(20)
    for x, y in state:
        print(ssm.predict(x), y)
