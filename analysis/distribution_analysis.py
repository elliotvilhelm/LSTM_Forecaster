from config import LABEL_UP, NONE, LABEL_DOWN

def get_class_sum(y):
    up = y[y == LABEL_UP].sum()
    none = y[y == NONE].sum()
    down = y[y == LABEL_DOWN].sum()
    total = up + none + down
    return (up/total) * 100, (none/total) * 100, \
           (down/total) * 100
