'''A module for testing utils_venv.'''
from functools import partial

import pytest
import gym
from gym.spaces import Box, Dict
import numpy as np


from custom_envs.vectorize import SubprocVecEnv, ThreadVecEnv

VECENV_CLASSES = [SubprocVecEnv, ThreadVecEnv]
NUMBER_OF_PROCESSORS = 2


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


@pytest.mark.parametrize("vecenv_class", VECENV_CLASSES)
def test_vec_env_reset(vecenv_class):
    '''A test for vec_env classes reset method.'''
    envs = [partial(StubEnv) for _ in range(NUMBER_OF_PROCESSORS)]
    vec_env = vecenv_class(envs)
    states = vec_env.reset()
    assert len(states['test1d']) == NUMBER_OF_PROCESSORS


@pytest.mark.parametrize("vecenv_class", VECENV_CLASSES)
def test_vec_env_step(vecenv_class):
    '''A test for vec_env classes step method.'''
    test = StubEnv()
    envs = [partial(StubEnv) for _ in range(NUMBER_OF_PROCESSORS)]
    vec_env = vecenv_class(envs)
    terminal = False
    while not terminal:
        actions = [test.action_space.sample()
                   for _ in range(NUMBER_OF_PROCESSORS)]
        states, rewards, terminal, info = vec_env.step(actions)
        assert len(states['test1d']) == NUMBER_OF_PROCESSORS
        assert len(rewards) == NUMBER_OF_PROCESSORS
        assert len(terminal) == NUMBER_OF_PROCESSORS
        assert len(info) == NUMBER_OF_PROCESSORS
        terminal = np.all(terminal)
