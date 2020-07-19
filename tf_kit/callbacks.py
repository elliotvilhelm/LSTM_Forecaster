import io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime as dt

from data_collection.yfinance_collector import get_multi_df
from analysis.confusion_matrix import get_confusion_matrix, plot_confusion_matrix
from config import TICKERS

date = dt.now().strftime('%Y-%m-%d_%H:%M_%p')
VALIDATION_CB = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'checkpoints/{date}' + '_{epoch:03d}',
    monitor='val_f1_score', verbose=1, save_best_only=False,
    save_weights_only=False, mode='auto', save_freq='epoch'
)

ld = f"logs/{date}/"
TENSORBOARD_CB = tf.keras.callbacks.TensorBoard(log_dir=ld + "scalar")


class GetConfusion(tf.keras.callbacks.Callback):
    def __init__(self):
        super(GetConfusion, self).__init__()
        _, _, self.x_val, self.y_val = get_multi_df(TICKERS)
        ld_cm = ld + "image"
        self.file_writer_cm = tf.summary.create_file_writer(ld_cm)

    def on_epoch_end(self, epoch, logs=None):
        test_pred_raw = self.model.predict(self.x_val)
    
        test_pred = np.argmax(test_pred_raw, axis=1)
        
        cm = get_confusion_matrix(self.model, self.x_val, self.y_val)
        figure = plot_confusion_matrix(cm.numpy(), ["UP", "NONE", "DOWN"])
            
        buf = io.BytesIO()
        
        plt.savefig(buf, format='png')
        
        plt.close(figure)
        buf.seek(0)

        image = tf.image.decode_png(buf.getvalue(), channels=4)
        
        cm_image = tf.expand_dims(image, 0)
            
        with self.file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)
CONFUSION_CB = GetConfusion()

class GetWeights(tf.keras.callbacks.Callback):
    def __init__(self):
        super(GetWeights, self).__init__()
        self.weight_dict = {}

    def on_epoch_end(self, epoch, logs=None):

        for layer_i in range(len(self.model.layers)):
            w = self.model.layers[layer_i].get_weights()[0]
            b = self.model.layers[layer_i].get_weights()[1]
            print('Layer %s has weights of shape %s and biases of shape %s' %(
                layer_i, np.shape(w), np.shape(b)))
            print(w)
            if epoch == 0:
                self.weight_dict['w_'+str(layer_i+1)] = w
                self.weight_dict['b_'+str(layer_i+1)] = b
            else:
                self.weight_dict['w_'+str(layer_i+1)] = np.dstack(
                    (self.weight_dict['w_'+str(layer_i+1)], w))
                self.weight_dict['b_'+str(layer_i+1)] = np.dstack(
                    (self.weight_dict['b_'+str(layer_i+1)], b)) 
WEIGHTS_CB = GetWeights()