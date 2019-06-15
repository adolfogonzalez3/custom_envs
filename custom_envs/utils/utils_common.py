'''
Module that contains common functions and classes for the package.
'''
from itertools import zip_longest, cycle, chain
from collections import deque

import numpy as np
import numpy.random as npr


def shuffle(*args, np_random=npr):
    '''
    Shuffle list-like args while maintaining row wise relationships.

    :param args: *[list or numpy.array] A variable number of arguments that
                                        are either lists or numpy arrays.
    :param np_random: (np.random.random_gen) A numpy random number generator.
    '''
    length = len(args[0])
    indices = np.arange(length)
    np_random.shuffle(indices)
    return [arg[indices] for arg in args]


def range_slice(low_or_high, high=None, step=32):
    '''
    Produce slices within a range of low to high with a specified stepsize.

    Each slice is non-overlapping.

    :param low_or_high: (int) An integer that specifies the low point or high
                              point of the range depending on the value given
                              to high.
    :param high: (int or None) If an integer, then specifies the high point of
                               the range. Otherwise if None then low_or_high
                               specifies the highest point of the range and
                               the low point of the range is zero.
    :param step: (int) The stepsize represents the size of each slice.
    '''
    if high is None:
        high = low_or_high
        low_or_high = 0
    low_iter = range(low_or_high, high, step)
    high_iter = range(low_or_high+step, high, step)
    return (slice(low, high) for low, high in zip_longest(low_iter, high_iter))


def batchify_zip(*args, size=32):
    '''
    Return an iterator that returns batches of each arg.

    Each batch is at most the size given.

    :param args: *[list or numpy.array] A variable number of arguments that
                                        are either lists or numpy arrays.
    :param size: (int) The size of each batch.
    :yield: ([numpy.array]) A list of numpy arrays.
    '''
    for batch_slice in range_slice(len(args[0]), step=size):
        yield [arg[batch_slice] for arg in args]


def ravel_zip(*args):
    '''
    Return an iterator that returns tuples of flatten arrays.

    :param args: *[list or numpy.array] A variable number of arguments that
                                        are numpy arrays.
    :param size: (int) The size of each batch.
    :yield: ([numpy.array]) A list of numpy arrays.
    '''
    for args_tuple in zip(*args):
        yield [arg.ravel() for arg in args_tuple]


def enzip(*iterables):
    '''
    Make an iterator that yields the index and the aggregates of the args.

    :param iterables: *[iterable] A variable number of arguments
                                             that are iterable.
    '''
    for i, iter_tuple in enumerate(zip(*iterables)):
        yield (i,) + iter_tuple


def to_onehot(array, num_of_labels=None):
    '''
    Convert a one dimensional array of catergorical values to a one hot array.

    :param array: (numpy.array) The array to convert to one hot.
    '''
    unique_array, array = np.unique(array, return_inverse=True)
    if num_of_labels is None:
        num_of_labels = unique_array.size
    onehot = np.zeros((len(array), num_of_labels))
    onehot[np.arange(len(array)), array] = 1
    return onehot, num_of_labels


class History:
    '''A class for storing a history of arrays.'''

    def __init__(self, max_history, **named_shapes):
        '''
        Create a History object.

        :param max_history: (int) The maximum number of past arrays stored.
        :param named_shapes: (**kwargs) Any arrays which are to be tracked.
        '''
        self.max_history = max_history
        self.shapes = {name: tuple(shape) if shape else (1,)
                       for name, shape in named_shapes.items()}
        self.history = {key: deque([np.zeros(shape)]*self.max_history,
                                   maxlen=self.max_history)
                        for key, shape in self.shapes.items()}
        self.iteration = 0

    def __getitem__(self, key):
        '''
        Get an item from history.

        :param key: (hashable) Any key able to be used by a python
                               dictionary.
        :return: (numpy.array) The requested item.
        '''
        return np.asarray(list(reversed(self.history[key])))

    def reset(self):
        '''Reset the history.'''
        self.history = {key: deque([np.zeros(shape)]*self.max_history,
                                   maxlen=self.max_history)
                        for key, shape in self.shapes.items()}
        self.iteration = 0

    def append(self, **named_items):
        '''
        Enqueue the named items in their appropiate places.

        :param named_items: (**kwargs) Should contain all keys in the
                                       dictionary.
        '''
        assert self.history.keys() == named_items.keys()
        for name, item in named_items.items():
            self.history[name].append(np.reshape(item, self.shapes[name]))
        self.iteration = (self.iteration + 1) % self.max_history

    def build_multistate(self):
        '''Build the state for multiple agents.'''
        shape = (self.max_history, -1)
        states = [self[key].reshape(shape).tolist()
                  for key in self.history.keys()]
        states = list(chain.from_iterable(states))
        states = [cycle(state) if len(state) == 1 else state
                  for state in states]
        states = list(zip(*states))
        return states


def build_multistate(grad_history, weight_history, loss_history, version=1):
    '''Build the state for multiple agents.'''
    if version == 0:
        states = [(g,) for g in grad_history[0].ravel().tolist()]
    elif version == 1:
        loss_flat = float(loss_history[0])
        grad_flat = grad_history[0].ravel().tolist()
        states = [(loss_flat, g) for g in grad_flat]
    elif version == 2:
        loss_flat = float(loss_history[0])
        grad_flat = grad_history[0].ravel().tolist()
        wght_flat = weight_history[0].ravel().tolist()
        states = [(w, loss_flat, g) for w, g in zip(wght_flat, grad_flat)]
    elif version == 3:
        shape = (len(loss_history), -1)
        loss_flat = tuple(loss_history.ravel().tolist())
        grad_flat = list(zip(*grad_history.reshape(shape).tolist()))
        wght_flat = list(zip(*weight_history.reshape(shape).tolist()))
        states = [w + loss_flat + g for w, g in zip(wght_flat, grad_flat)]
    return states
