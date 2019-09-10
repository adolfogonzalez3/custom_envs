'''A module for testing InMemoryDataSet.'''

import numpy as np
import numpy.random as npr

from custom_envs.dataset import InMemoryDataSet


def random_data_set(sample_shape=(10,), num_of_targets=1, batch_size=None):
    features = npr.rand(*sample_shape)
    targets = npr.rand(sample_shape[0], num_of_targets)
    return InMemoryDataSet(features, targets, batch_size=batch_size)


def test_on_epoch_end():
    '''Test on_epoch_end method.'''
    features = npr.rand(10)
    targets = npr.rand(10)
    data_set = InMemoryDataSet(features, targets)
    assert np.all(features == data_set.features)
    assert np.all(targets == data_set.targets)
    data_set.on_epoch_end()
    assert not np.all(features == data_set.features)
    assert not np.all(targets == data_set.targets)


def test_len():
    '''Test __len__ method.'''
    data_set = random_data_set(sample_shape=(10,))
    assert len(data_set) == 1
    data_set = random_data_set(sample_shape=(10,), batch_size=2)
    assert len(data_set) == 5


def test_getitem():
    '''Test __getitem__ method.'''
    data_set = random_data_set(sample_shape=(10,))
    assert len(data_set[0].features) == 10
    data_set = random_data_set(sample_shape=(10,), batch_size=2)
    assert len(data_set[0].features) == 2


def test_feature_shape():
    '''Test feature_shape property.'''
    data_set = random_data_set()
    assert data_set.feature_shape == ()


def test_target_shape():
    '''Test target_shape property.'''
    data_set = random_data_set()
    assert data_set.target_shape == (1,)
