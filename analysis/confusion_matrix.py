import tensorflow as tf
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
from config import N_CLASSES
mpl.use('Agg')


def log_class_distributions(y_l, pred_l):
    y_class_distributions = []
    p_class_distributions = []
    for n in range(N_CLASSES):
        y_class_distributions.append(round((y_l.count(n) / len(y_l)) * 100, 2))
        p_class_distributions.append(round((pred_l.count(n) / len(y_l)) * 100, 2))

    print('-' * 80)
    print("[CONFUSION STATS]")
    for n in range(N_CLASSES):
        print(f"class_{n} y pct: {y_class_distributions[n]}%")
    for n in range(N_CLASSES):
        print(f"class_{n} prediction pct: {p_class_distributions[n]}%")
    print('-' * 80)


def get_confusion_matrix(model, x, y):
    """
    obtain confusion matrix from multiclass model. 
    """
    pred_l = []
    pred = model.predict(x)
    for p in pred:
        pred_l.append(p.argmax())
    y_l = []
    for y in y:
        y_l.append(y.argmax())

    log_class_distributions(y_l, pred_l)

    return tf.math.confusion_matrix(y_l, pred_l)


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", va="center", bbox={'alpha':1,'edgecolor':'none','pad':1}, color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    return figure
