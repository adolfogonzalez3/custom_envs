'''Module which holds math functions.'''
from contextlib import contextmanager

import numexpr
import numpy as np
import numpy.random as npr


@contextmanager
def use_random_state(random_state):
    '''
    Set the random state for the current context.

    :param random_state: (numpy.random.RandomState) The random state generator
                                                    to use for the context.
    '''
    saved_state = npr.get_state()
    try:
        npr.set_state(random_state.get_state())
        yield random_state
    finally:
        npr.set_state(saved_state)


def cross_entropy(prob, ground_truth):
    '''
    Compute the cross entropy loss between predictions and ground truths.

    :param prob: (numpy.array) An array of predictions of similar shape to
                               ground_truth.
    :param ground_truth: (numpy.array) An array of ground truths.
    '''
    prob_log = np.log(prob+1e-16)
    return np.mean(np.sum(-prob_log*ground_truth, axis=1))


def mse(prediction, ground_truth):
    '''
    Compute the mean squared error loss between predictions and ground truths.

    :param prediction: (numpy.array) An array of predictions of similar shape
                                     to ground_truth.
    :param ground_truth: (numpy.array) An array of ground truths.
    '''
    result = numexpr.evaluate('sum((prediction - ground_truth)**2, axis=1)',
                              local_dict={'prediction': prediction,
                                          'ground_truth': ground_truth})
    return np.mean(result/2)


def softmax(logits):
    '''
    Compute softmax on an array of values.

    :param x: A two dimensional array of values where values on the same row
              have softmax applied to them.
    '''
    logits_max = np.max(logits, axis=1)[:, None]
    p_exp = numexpr.evaluate('exp(logits - logits_max)',
                             local_dict={'logits': logits,
                                         'logits_max': logits_max})
    p_sum = np.sum(p_exp, axis=1)[:, None]
    return p_exp/p_sum


def sigmoid(logits):
    '''
    Compute sigmoid on an array of values.

    :param x: A two dimensional array of values where values on the same row
              have sigmoid applied to them.
    '''
    return numexpr.evaluate('1 / (1 + exp(-logits - 1e-8))',
                            local_dict={'logits': logits})


def normalize(data):
    '''
    Normalize the data to be within 0. and 1.

    :param data: The data to normalize.
    '''
    mins = np.min(data, axis=0)
    maxes = np.max(data, axis=0)
    return numexpr.evaluate('(data - mins) / (maxes - mins + 1e-8)',
                            local_dict={'mins': mins, 'maxes': maxes,
                                        'data': data})
