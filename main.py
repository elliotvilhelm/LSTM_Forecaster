import tensorflow as tf
from data_processing.data_processing import get_datasets, add_features, get_tfds
from market.Equity import EquityData
from tf_kit.callbacks import TENSORBOARD_CB, VALIDATION_CB
from tf_kit.model import get_lstm
from config import BATCH_SIZE, EPOCHS, DATA_DIR, DATA_SYM
from analysis.confusion_matrix import get_confusion_matrix

tf.random.set_seed(42)

if __name__ == "__main__":
    e = EquityData(DATA_DIR, DATA_SYM)
    e = add_features(e)
    x_train, y_train, x_val, y_val = get_datasets(e)
    tfds_train, tfds_val = get_tfds(x_train, y_train, x_val, y_val)
    window = int(e.data.shape[0] / BATCH_SIZE)

    ssm = get_lstm()

    history = ssm.fit(tfds_train, epochs=EPOCHS,
                      steps_per_epoch=window,
                      validation_data=tfds_val,
                      validation_steps=20,
                      callbacks=[VALIDATION_CB, TENSORBOARD_CB])

    for x, y in tfds_val.take(10):
        print(ssm.predict(x), y)

    get_confusion_matrix(ssm, x_val, y_val)

