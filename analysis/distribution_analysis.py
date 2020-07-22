from config import LABEL_UP, NONE, LABEL_DOWN


def get_class_sum(y):
    up = y[y == LABEL_UP].sum()
    none = y[y == NONE].sum()
    down = y[y == LABEL_DOWN].sum()
    total = up + none + down
    return (up/total) * 100, (none/total) * 100, \
           (down/total) * 100


def log_distributions(x_train, y_train, x_val, y_val):
    print("-" * 80)
    print("CLASS DISTRIBUTIONS:\n")
    print("UP\tNONE\tDOWN")
    print(["{:2}%".format(round(x, 2)) for x in get_class_sum(y_train)])
    print(["{:2}%".format(round(x, 2)) for x in get_class_sum(y_val)])
    print(f"X train: {x_train.shape}\n"
          f"y train: {y_train.shape}\n"
          f"X val: {x_val.shape}\n"
          f"y val: {y_val.shape}")
    print("-" * 80)

