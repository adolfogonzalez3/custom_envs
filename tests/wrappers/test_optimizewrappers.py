'''A module that tests optimizewrappers.'''
import numpy as np
import numpy.random as npr

import gym
from gym.spaces import Box, Dict

import custom_envs.wrappers.optimizewrappers as wrappers


class StubEnv(gym.Env):
    '''Environment for testing.'''

    def __init__(self):
        self.counter = 0
        self.observation_space = Dict({
            'test1d': Box(low=-1e3, high=1e3, dtype=np.float32, shape=[5]),
            'test2d': Box(low=-1e3, high=1e3, dtype=np.float32, shape=[5]*2),
            'test3d': Box(low=-1e3, high=1e3, dtype=np.float32, shape=[5]*3)
        })
        self.action_space = Box(low=-1e3, high=1e3, dtype=np.float32,
                                shape=(25,))
        self.observation_space_old = self.observation_space
        self.action_space_old = self.action_space

    def step(self, action):
        '''Stub.'''
        self.counter += 1
        return self.observation_space_old.sample(), 0, self.terminal(), {}

    def terminal(self):
        '''Stub.'''
        return self.counter >= 10

    def render(self, mode='human'):
        '''Stub.'''

    def reset(self):
        '''Stub.'''
        self.counter = 0
        return self.observation_space_old.sample()


def test_historywrapper_spaces():
    '''Test HistoryWrapper's spaces.'''
    max_history = 5
    env = StubEnv()
    obs_space_old = env.observation_space.spaces
    action_space_old = env.action_space
    env = wrappers.HistoryWrapper(env, max_history)
    assert obs_space_old.keys() == env.observation_space.spaces.keys()
    for name, space in obs_space_old.items():
        assert np.all(env.observation_space[name].low == space.low)
        assert np.all(env.observation_space[name].high == space.high)
        assert env.observation_space[name].shape == (max_history, *space.shape)
    assert action_space_old == env.action_space


def test_historywrapper_step():
    '''Test HistoryWrapper's step method.'''
    max_history = 5
    env = wrappers.HistoryWrapper(StubEnv(), max_history)
    obs, reward, terminal, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, (float, int))
    assert isinstance(terminal, bool)
    assert isinstance(info, dict)


def test_historywrapper_reset():
    '''Test HistoryWrapper's reset method.'''
    max_history = 5
    env = wrappers.HistoryWrapper(StubEnv(), max_history)
    obs = env.reset()
    assert env.observation_space.contains(obs)


def test_subsetwrapper_spaces():
    '''Test SubSetWrapper's spaces.'''
    env = StubEnv()
    obs_space_old = env.observation_space
    action_space_old = env.action_space
    env = wrappers.SubSetWrapper(env, ['test1d', 'test2d'])
    assert obs_space_old.spaces.keys() >= env.observation_space.spaces.keys()
    for name, space in env.observation_space.spaces.items():
        assert np.all(obs_space_old[name].low == space.low)
        assert np.all(obs_space_old[name].high == space.high)
        assert obs_space_old[name].shape == space.shape
    assert action_space_old == env.action_space


def test_subsetwrapper_step():
    '''Test SubSetWrapper's step method.'''
    env = wrappers.SubSetWrapper(StubEnv(), ['test1d', 'test2d'])
    obs, reward, terminal, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, (float, int))
    assert isinstance(terminal, bool)
    assert isinstance(info, dict)


def test_subsetwrapper_reset():
    '''Test SubSetWrapper's reset method.'''
    env = wrappers.SubSetWrapper(StubEnv(), ['test1d', 'test2d'])
    obs = env.reset()
    assert env.observation_space.contains(obs)
