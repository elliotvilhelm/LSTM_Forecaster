from config import LABEL_UP, LABEL_DOWN, LABEL_UP_CHOP, LABEL_DOWN_CHOP

def get_class_sum(y):
    up = y[y == LABEL_UP].sum()
    up_chop = y[y == LABEL_UP_CHOP].sum()
    down = y[y == LABEL_DOWN].sum()
    down_chop = y[y == LABEL_DOWN_CHOP].sum()
    total = up + up_chop + down_chop + down_chop
    return (up/total) * 100, (up_chop/total) * 100, \
           (down/total) * 100, (down_chop/total) * 100
