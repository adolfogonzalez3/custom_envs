'''Tests utils_common module.'''
import pytest
import numpy as np
import numpy.random as npr

import custom_envs.utils.utils_common as utils

BATCH_SIZES = tuple(2**np.arange(5))
MAGNITUDE = tuple(range(5))
NUM_OF_SEED = 5
NUM_OF_ARRAYS = 5
SHAPES = tuple((2**i, i + 2) for i in range(5))


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


@pytest.mark.parametrize("seed", range(NUM_OF_SEED))
@pytest.mark.parametrize("shape", SHAPES)
def test_cross_entropy(seed, shape):
    '''Tests cross_entropy.'''
    npr.seed(seed)
    array = utils.softmax(npr.uniform(size=shape))
    labels = npr.uniform(size=shape)
    labels = labels / np.expand_dims(np.mean(labels, axis=1), axis=1)
    cost = utils.cross_entropy(array, labels)
    assert cost >= 0
    assert isinstance(cost, float)


@pytest.mark.parametrize("seed", range(NUM_OF_SEED))
@pytest.mark.parametrize("shape", SHAPES)
def test_mse(seed, shape):
    '''Tests mse.'''
    npr.seed(seed)
    array = npr.uniform(size=shape)
    array = array - np.min(array) / np.ptp(array)
    labels = npr.uniform(size=shape)
    cost = utils.mse(array, labels)
    assert cost >= 0
    assert isinstance(cost, float)


@pytest.mark.parametrize("seed", range(NUM_OF_SEED))
@pytest.mark.parametrize("magnitude", MAGNITUDE)
@pytest.mark.parametrize("shape", SHAPES)
def test_softmax(seed, magnitude, shape):
    '''Tests softmax.'''
    npr.seed(seed)
    low = -10**magnitude
    high = 10**magnitude
    array = npr.uniform(low, high, size=shape)
    array = utils.softmax(array)
    summation = np.sum(array, axis=1)
    assert np.all(array >= 0)
    assert np.all(array <= 1)
    assert np.all(np.isclose(summation, 1))


@pytest.mark.parametrize("seed", range(NUM_OF_SEED))
@pytest.mark.parametrize("magnitude", MAGNITUDE)
@pytest.mark.parametrize("shape", SHAPES)
def test_sigmoid(seed, magnitude, shape):
    '''Tests sigmoid.'''
    npr.seed(seed)
    low = -10**magnitude
    high = 10**magnitude
    array = npr.randint(low, high, size=shape)
    array = utils.sigmoid(array)
    assert np.all(array >= 0)
    assert np.all(array <= 1)


@pytest.mark.parametrize("size", range(10, 100, 10))
@pytest.mark.parametrize("num_of_labels", range(2, 10))
def test_to_onehot(size, num_of_labels):
    '''Test to_onehot.'''
    array = np.arange(size) % num_of_labels
    onehot, num_onehot_labels = utils.to_onehot(array)
    assert num_onehot_labels == num_of_labels
    assert onehot.shape == (size, num_of_labels)
    assert len(np.unique(onehot, axis=0)) == num_of_labels


@pytest.mark.parametrize("seed", range(NUM_OF_SEED))
@pytest.mark.parametrize("shape", SHAPES)
def test_normalize_bound(seed, shape):
    '''Test that normalize function normalizes within 0 and 1.'''
    npr.seed(seed)
    array = npr.randn(*shape)
    array = utils.normalize(array)
    assert np.all(array >= 0)
    assert np.all(array <= 1)


@pytest.mark.parametrize("seed", range(NUM_OF_SEED))
@pytest.mark.parametrize("samples", range(10, 100, 10))
def test_use_random_state(seed, samples):
    '''Tests context manager use_random_state.'''
    random_state = npr.RandomState(seed)
    with utils.use_random_state(random_state):
        test_context = tuple(npr.rand() for _ in range(samples))
    random_state = npr.RandomState(seed)
    test = tuple(random_state.rand() for _ in range(samples))
    assert test_context == test


def test_ravel_zip():
    '''Tests ravel_zip.'''
    arrays = [npr.rand(128, 2**i) for i in range(10)]
    for arrays_flat in utils.ravel_zip(*arrays):
        assert sum(a.size for a in arrays_flat) == (2**10 - 1)


def test_enzip():
    '''Tests enzip.'''
    for i, j, _ in utils.enzip(range(10), range(100)):
        assert i == j
