import tensorflow as tf


def get_confusion_matrix(model, x, y):
    pred_l = []
    pred = model.predict(x)
    for p in pred:
        pred_l.append(p.argmax())
    y_l = []
    for y in y:
        y_l.append(y.argmax())

    return tf.math.confusion_matrix(y_l, pred_l)
