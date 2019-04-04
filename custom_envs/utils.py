
from itertools import zip_longest

import numpy as np
import numpy.random as npr
import numexpr


def shuffle(*args, np_random=npr):
    length = len(args[0])
    indices = np.arange(length)
    np_random.shuffle(indices)
    return [arg[indices] for arg in args]

def range_slice(low_or_high, high=None, step=32):
    if high is None:
        high = low_or_high
        low_or_high = 0
    low_iter = range(low_or_high, high, step)
    high_iter = range(low_or_high+step, high, step)
    return (slice(low, high) for low, high in zip_longest(low_iter, high_iter))

def batchify_zip(*args, size=32):
    for batch_slice in range_slice(len(args[0]), step=size):
        yield [arg[batch_slice] for arg in args]

def cross_entropy(p, y):
    #p_log = np.nan_to_num(np.log(p))
    p_log = np.log(p+1e-16)
    return np.mean(np.sum(-p_log*y, axis=1))
    
def mse(p, y):
    return np.mean(np.sum((p - y)**2, axis=1)/2)

def softmax(x):
    #x = (x - np.max(x))/(np.max(x) - np.min(x) + 1e-8)
    p_exp = np.exp(x - np.max(x, axis=1)[:, None])
    #p_exp = np.exp(x)
    p_sum = np.sum(p_exp, axis=1)
    return p_exp/p_sum[:, None]

def sigmoid(x):
    return numexpr.evaluate('1 / (1 + exp(-x - 1e-8))')

def to_onehot(x):
    x = x.astype(np.int).ravel()
    num_of_labels = np.unique(x).size
    onehot = np.zeros((len(x), num_of_labels))
    onehot[np.arange(len(x)), x] = 1
    return onehot, num_of_labels
