import numpy as np
import tensorflow as tf

def create_time_steps(length):
    return list(range(-length, 0))
