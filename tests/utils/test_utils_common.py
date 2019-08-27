'''Tests utils_common module.'''
from itertools import chain

import pytest
import numpy as np
import numpy.random as npr

import custom_envs.utils.utils_common as utils

BATCH_SIZES = tuple(2**np.arange(3))
NUM_OF_SEED = 3
NUM_OF_ARRAYS = 3


@pytest.mark.parametrize("seed", range(NUM_OF_SEED))
def test_shuffle(seed):
    '''Tests shuffle function for alignment.'''
    npr.seed(seed)
    array_a = np.arange(25, 0, -1)
    array_b = np.arange(25, 0, -1)
    a_shuffle, b_shuffle = utils.shuffle(array_a, array_b)
    assert np.all(a_shuffle == b_shuffle)


def test_range_slice():
    '''Tests range_slice for correct slicing.'''
    array = np.arange(256)
    for i, batch in enumerate(utils.range_slice(0, 256, 32)):
        assert np.all(array[batch] == array[i*32:(i+1)*32])


@pytest.mark.parametrize("num_of_arrays", range(1, NUM_OF_ARRAYS))
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_batchify_zip(num_of_arrays, batch_size):
    '''Tests for batchify_zip.'''
    arrays = [np.arange(batch_size*i, batch_size*(i+1))
              for i in range(num_of_arrays)]
    for i, batch in enumerate(utils.batchify_zip(*arrays, size=batch_size)):
        for array, array_batch in zip(arrays, batch):
            assert len(array_batch) == batch_size
            assert np.all(array_batch == array[i*batch_size:batch_size*(i+1)])


@pytest.mark.parametrize("size", range(10, 100, 10))
@pytest.mark.parametrize("num_of_labels", range(2, 10))
def test_to_onehot(size, num_of_labels):
    '''Test to_onehot.'''
    array = np.arange(size) % num_of_labels
    onehot, num_onehot_labels = utils.to_onehot(array)
    assert num_onehot_labels == num_of_labels
    assert onehot.shape == (size, num_of_labels)
    assert len(np.unique(onehot, axis=0)) == num_of_labels


def test_ravel_zip():
    '''Tests ravel_zip.'''
    arrays = [npr.rand(128, 2**i) for i in range(10)]
    for arrays_flat in utils.ravel_zip(*arrays):
        assert sum(a.size for a in arrays_flat) == (2**10 - 1)


def test_enzip():
    '''Tests enzip.'''
    for i, j, _ in utils.enzip(range(10), range(100)):
        assert i == j


def test_history_build_multistate_version():
    '''Test history's build_multistate method.'''
    test_a = npr.rand(5, 5)
    test_b = npr.rand(1)
    history = utils.History(1, test_a=(5, 5), test_b=(1,))
    history.append(test_a=test_a, test_b=test_b)
    hist_a, hist_b = zip(*history.build_multistate())
    assert np.all(hist_a == test_a.ravel())
    assert np.all(hist_b == test_b.ravel())


def test_history_getitem():
    '''Test History's __getindex__ method.'''
    max_history = 3
    history = utils.History(max_history, test1d=(5,), test2d=(5, 5))
    assert history['test1d'].shape == (max_history, 5)
    assert history['test2d'].shape == (max_history, 5, 5)


def test_history_iter():
    '''Test History's __iter__ method.'''
    max_history = 3
    test = {chr(97+i): (i+1,) for i in range(26)}
    history = utils.History(max_history, **test)
    for key, history_key in zip(test, history):
        assert key == history_key


def test_history_len():
    '''Test History's __len__ method.'''
    max_history = 3
    test = {chr(97+i): (i+1,) for i in range(26)}
    history = utils.History(max_history, **test)
    assert len(history) == len(test)


def test_history_append():
    '''Test History's append method.'''
    max_history = 3
    history = utils.History(max_history, test1d=(5,), test2d=(5, 5))
    test1d = np.arange(max_history*5).reshape((max_history, 5))
    test2d = np.arange(max_history*25).reshape((max_history, 5, 5))
    for i in range(max_history):
        history.append(test1d=test1d[-(i+1)],
                       test2d=test2d[-(i+1)])
    assert np.all(history['test1d'] == test1d)
    assert np.all(history['test2d'] == test2d)


def test_history_reset():
    '''Test History's reset method.'''
    max_history = 3
    history = utils.History(max_history, test1d=(5,), test2d=(5, 5))
    test1d = np.arange(max_history*5).reshape((max_history, 5))
    test2d = np.arange(max_history*25).reshape((max_history, 5, 5))
    for i in range(max_history):
        history.append(test1d=test1d[-(i+1)],
                       test2d=test2d[-(i+1)])
    history.reset()
    assert np.all(history['test1d'] == 0)
    assert np.all(history['test2d'] == 0)
    history.reset(test1d=np.ones((5,)), test2d=np.ones((5, 5)))
    assert np.all(history['test1d'] == 1)
    assert np.all(history['test2d'] == 1)


def test_flatten_arrays():
    '''Test flatten_arrays function.'''
    arrays = [npr.rand(i+1, i+1) for i in range(10)]
    total_size = sum(a.size for a in arrays)
    flattened_array = utils.flatten_arrays(arrays)
    assert flattened_array.size == total_size
    arrays = [a.ravel() for a in arrays]
    for num, fnum in zip(chain.from_iterable(arrays), flattened_array):
        assert num == fnum


def test_from_flat():
    '''Test from_flat function.'''
    shapes = [(i+1, i+1) for i in range(10)]
    arrays = [npr.rand(*shape) for shape in shapes]
    flattened_array = utils.flatten_arrays(arrays)
    reshaped_arrays = utils.from_flat(flattened_array, shapes)
    assert len(arrays) == len(reshaped_arrays)
    for array, rarray in zip(arrays, reshaped_arrays):
        assert np.all(array == rarray)


def test_addprogressbar():
    '''Test AddProgressBar class.'''
    iterator = iter(range(100))
    some_list = utils.AddProgressBar([], ['append'], iterator)
    for i in range(5):
        some_list.append(i)
    assert next(iterator) == 5
