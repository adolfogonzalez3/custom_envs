'''
Module that contains common utilities for the package.
'''

from pathlib import Path
from itertools import zip_longest
from contextlib import contextmanager

import gym
import numexpr
import numpy as np
import numpy.random as npr

from custom_envs.utils.utils_logging import Monitor


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


def create_env(env_name, log_dir=None, num_of_envs=1, **kwarg):
    '''
    Create many environments at once with optional independent logging.

    :param env_name: (str) The name of an OpenAI environment to create.
    :param log_dir: (str or None) If str then log_dir is the path to save all
                                  log files to otherwise if None then don't
                                  log anything.
    :param num_of_envs: (int) The number of environments to create.
    '''
    envs = [gym.make(env_name, **kwarg) for _ in range(num_of_envs)]
    if log_dir is not None:
        log_dir = Path(log_dir)
        envs = [Monitor(env, str(log_dir / str(i)), allow_early_resets=True,
                        info_keywords=('objective', 'accuracy'),
                        chunk_size=10)
                for i, env in enumerate(envs)]
    return envs


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
