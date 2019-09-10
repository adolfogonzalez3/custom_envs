'''Tests utils_env module.'''

import gym
import pytest
import numpy as np
import numpy.random as npr

import custom_envs.utils.utils_common as utils_common
import custom_envs.utils.utils_env as utils


def create_random_history(shape=(5, 5), max_history=10):
    '''
    Create a history object with random elements.

    :param max_history: (int) The max history to store.
    :return: (History) A History object.
    '''
    history = utils_common.History(max_history, weights=shape, gradients=shape,
                                   losses=())
    for _ in range(max_history):
        history.append(weights=npr.rand(*shape), losses=npr.rand(),
                       gradients=npr.rand(*shape))
    return history


def test_get_observation_space_and_history_v0():
    '''Test the get_observation_space_and_history function version 0.'''
    shape = (5, 5)
    max_history = 10
    space, hist = utils.get_obs_version(shape, max_history, version=0)
    assert isinstance(space, gym.spaces.Box)
    assert isinstance(hist, utils_common.History)
    assert hist.max_history == 1
    assert {'gradients'} == hist.keys()
    assert space.shape == (1,)
    assert np.all(space.low == -1e6)
    assert np.all(space.high == 1e6)
    assert space.dtype == np.float32


def test_get_observation_space_and_history_v1():
    '''Test the get_observation_space_and_history function version 1.'''
    shape = (5, 5)
    max_history = 10
    space, hist = utils.get_obs_version(shape, max_history, version=1)
    assert isinstance(space, gym.spaces.Box)
    assert isinstance(hist, utils_common.History)
    assert hist.max_history == max_history
    assert {'losses', 'gradients'} == hist.keys()
    assert space.shape == (2*max_history,)
    assert np.all(space.low == -1e6)
    assert np.all(space.high == 1e6)
    assert space.dtype == np.float32


def test_get_observation_space_and_history_v2():
    '''Test the get_observation_space_and_history function version 2.'''
    shape = (5, 5)
    max_history = 10
    space, hist = utils.get_obs_version(shape, max_history, version=2)
    assert isinstance(space, gym.spaces.Box)
    assert isinstance(hist, utils_common.History)
    assert hist.max_history == 1
    assert {'weights', 'losses', 'gradients'} == hist.keys()
    assert space.shape == (3,)
    assert np.all(space.low == -1e6)
    assert np.all(space.high == 1e6)
    assert space.dtype == np.float32


def test_get_observation_space_and_history_v3():
    '''Test the get_observation_space_and_history function version 3.'''
    shape = (5, 5)
    max_history = 10
    space, hist = utils.get_obs_version(shape, max_history, version=3)
    assert isinstance(space, gym.spaces.Box)
    assert isinstance(hist, utils_common.History)
    assert hist.max_history == max_history
    assert {'weights', 'losses', 'gradients'} == hist.keys()
    assert space.shape == (3*max_history,)
    assert np.all(space.low == -1e6)
    assert np.all(space.high == 1e6)
    assert space.dtype == np.float32


def test_get_observation_space_and_history_v4():
    '''Test the get_observation_space_and_history function version 4.'''
    shape = (5, 5)
    max_history = 10
    space, hist = utils.get_obs_version(shape, max_history, version=4)
    assert isinstance(space, gym.spaces.Box)
    assert isinstance(hist, utils_common.History)
    assert hist.max_history == max_history
    assert {'gradients'} == hist.keys()
    assert space.shape == (max_history,)
    assert np.all(space.low == -1e6)
    assert np.all(space.high == 1e6)
    assert space.dtype == np.float32


def test_get_observation_space_and_history_not_a_version():
    '''
    Test the get_observation_space_and_history function with invalid version.
    '''
    shape = (5, 5)
    max_history = 10
    with pytest.raises(RuntimeError):
        utils.get_obs_version(shape, max_history, version=-1)


def test_get_action_space_optlrs_v0():
    '''Test get_action_space_optlrs function version 0.'''
    space = utils.get_action_space_optlrs(version=0)
    assert isinstance(space, gym.spaces.Box)
    assert np.all(space.low == -4.)
    assert np.all(space.high == 6.)
    assert space.dtype == np.float32
    assert space.shape == (1,)


def test_get_action_space_optlrs_v1():
    '''Test get_action_space_optlrs function version 1.'''
    space = utils.get_action_space_optlrs(version=1)
    assert isinstance(space, gym.spaces.Box)
    assert np.all(space.low == 0.)
    assert np.all(space.high == 1e4)
    assert space.dtype == np.float32
    assert space.shape == (1,)


def test_get_action_space_optlrs_not_a_version():
    '''Test get_action_space_optlrs function with invalid version.'''
    with pytest.raises(RuntimeError):
        utils.get_action_space_optlrs(version=-1)


def test_get_reward_v0():
    '''Test the get_reward function version 0.'''
    loss = 1e2
    adjusted_loss = .8
    reward = utils.get_reward(loss, adjusted_loss, version=0)
    assert isinstance(reward, float)
    assert reward < 0


def test_get_reward_v1():
    '''Test the get_reward function version 1.'''
    loss = 1e2
    adjusted_loss = .8
    reward = utils.get_reward(loss, adjusted_loss, version=1)
    assert isinstance(reward, float)
    assert reward > 0


def test_get_reward_not_a_version():
    '''Test the get_reward function with invalid version.'''
    loss = 1e2
    adjusted_loss = .8
    with pytest.raises(RuntimeError):
        utils.get_reward(loss, adjusted_loss, version=-1)


def test_get_action_optlrs_v0():
    '''Test the get_action_optlrs version 0.'''
    action = npr.normal(size=(5, 5))
    new_action = utils.get_action_optlrs(action, version=0)
    assert np.all(new_action > 0)


def test_get_action_optlrs_v1():
    '''Test the get_action_optlrs version 1.'''
    action = npr.uniform(0, 1e3, size=(5, 5))
    new_action = utils.get_action_optlrs(action, version=1)
    assert np.all(new_action >= 0)


def test_get_action_optlrs_not_a_verion():
    '''Test the get_action_optlrs version with invalid version.'''
    action = npr.uniform(0, 1e3, size=(5, 5))
    with pytest.raises(RuntimeError):
        utils.get_action_optlrs(action, version=-1)


def test_get_observation_v0():
    '''Test the get_observation function with version 0.'''
    history = create_random_history()
    loss, grad, weights = utils.get_observation(history, version=0)
    assert isinstance(loss, float)
    assert isinstance(grad, np.ndarray)
    assert isinstance(weights, np.ndarray)
    assert len(grad) == len(weights)


def test_get_observation_v1():
    '''Test the get_observation function with version 1.'''
    history = create_random_history()
    loss, grad, weights = utils.get_observation(history, version=1)
    assert isinstance(loss, float)
    assert isinstance(grad, np.ndarray)
    assert isinstance(weights, np.ndarray)
    assert len(grad) == len(weights)


def test_get_observation_v2():
    '''Test the get_observation function with version 2.'''
    history = create_random_history()
    loss, grad, weights = utils.get_observation(history, version=2)
    assert isinstance(loss, float)
    assert isinstance(grad, np.ndarray)
    assert isinstance(weights, np.ndarray)
    assert len(grad) == len(weights)


def test_get_observation_v3():
    '''Test the get_observation function with version 3.'''
    history = create_random_history()
    loss, grad, weights = utils.get_observation(history, version=3)
    assert isinstance(loss, float)
    assert isinstance(grad, np.ndarray)
    assert isinstance(weights, np.ndarray)
    assert len(grad) == len(weights)


def test_get_observation_not_a_version():
    '''Test the get_observation function with invalid version.'''
    history = create_random_history()
    with pytest.raises(RuntimeError):
        loss, grad, weights = utils.get_observation(history, version=100)
