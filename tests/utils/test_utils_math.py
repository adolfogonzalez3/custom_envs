'''Tests utils_common module.'''
import pytest
import numpy as np
import numpy.random as npr

import custom_envs.utils.utils_math as utils

BATCH_SIZES = tuple(2**np.arange(3))
MAGNITUDE = tuple(range(3))
NUM_OF_SEED = 3
NUM_OF_ARRAYS = 3
SHAPES = tuple((2**i, i + 2) for i in range(3))


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
    array = npr.uniform(low, high, size=shape).tolist()
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


@pytest.mark.parametrize("seed", range(NUM_OF_SEED))
@pytest.mark.parametrize("shape", SHAPES)
def test_normalize_bound(seed, shape):
    '''Test that normalize function normalizes within 0 and 1.'''
    npr.seed(seed)
    array = npr.randn(*shape)
    array = utils.normalize(array)
    assert np.all(array >= 0)
    assert np.all(array <= 1)
