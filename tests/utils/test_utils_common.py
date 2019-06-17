'''Tests utils_common module.'''
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


def test_history_build_multistate_version_0():
    '''Test history's build_multistate method version 0.'''
    max_history = 3
    gradients = npr.rand(max_history, 5, 5)
    history = utils.History(1, gradients=(5, 5))
    history.append(gradients=gradients[0])
    func_states = utils.build_multistate(gradients, None, None, version=0)
    hist_states = history.build_multistate()
    assert func_states == hist_states


def test_history_build_multistate_version_1():
    '''Test history's build_multistate method version 1.'''
    max_history = 3
    loss = npr.rand(max_history, 1)
    gradients = npr.rand(max_history, 5, 5)
    history = utils.History(1, losses=(), gradients=(5, 5))
    history.append(losses=loss[0], gradients=gradients[0])
    func_states = utils.build_multistate(gradients, None, loss, version=1)
    hist_states = history.build_multistate()
    assert func_states == hist_states


def test_history_build_multistate_version_2():
    '''Test history's build_multistate method version 2.'''
    max_history = 3
    losses = npr.rand(max_history, 1)
    weights = npr.rand(max_history, 5, 5)
    gradients = npr.rand(max_history, 5, 5)
    history = utils.History(1, weights=(5, 5), losses=(), gradients=(5, 5))
    history.append(weights=weights[0], losses=losses[0],
                   gradients=gradients[0])
    func_states = utils.build_multistate(gradients, weights, losses, version=2)
    hist_states = history.build_multistate()
    assert func_states == hist_states


def test_history_build_multistate_version_3():
    '''Test history's build_multistate method version 2.'''
    max_history = 3
    losses = npr.rand(max_history, 1)
    weights = npr.rand(max_history, 5, 5)
    gradients = npr.rand(max_history, 5, 5)
    history = utils.History(max_history, weights=(5, 5), losses=(),
                            gradients=(5, 5))
    history.append(weights=weights[0], losses=losses[0],
                   gradients=gradients[0])
    for wght, loss, grad in zip(weights, losses, gradients):
        history.append(weights=wght, losses=loss, gradients=grad)
    gradients = np.array(list(reversed(gradients)))
    weights = np.array(list(reversed(weights)))
    losses = np.array(list(reversed(losses)))
    func_states = utils.build_multistate(gradients, weights, losses, version=3)
    hist_states = history.build_multistate()
    assert func_states == hist_states


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
