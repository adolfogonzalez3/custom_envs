'''A module for testing utils_venv.'''
from itertools import chain

import numpy as np
import gym
from gym.spaces import Box, Dict

from custom_envs.vectorize.optvecenv import OptVecEnv, flatten_dictionary


class StubEnv(gym.Env):
    '''Environment for testing.'''

    def __init__(self):
        self.counter = 0
        self.observation_space = Dict({
            'test1d': Box(low=-1e3, high=1e3, dtype=np.float32, shape=[5]),
            'test2d': Box(low=-1e3, high=1e3, dtype=np.float32, shape=[5]),
            'test3d': Box(low=-1e3, high=1e3, dtype=np.float32, shape=[5])
        })
        self.action_space = self.observation_space

    def step(self, action):
        '''Stub.'''
        self.counter += 1
        return self.observation_space.sample(), 0, self.terminal(), {}

    def terminal(self):
        '''Stub.'''
        return self.counter >= 10

    def render(self, mode='human'):
        '''Stub.'''

    def reset(self):
        '''Stub.'''
        self.counter = 0
        return self.observation_space.sample()


def test_optvecenv_reset():
    '''A test for vec_env classes reset method.'''
    vec_env = OptVecEnv([StubEnv])
    states = vec_env.reset()
    #assert len(states) == len(env.observation_space.spaces)


def test_optvecenv_step():
    '''A test for vec_env classes step method.'''
    vec_env = OptVecEnv([StubEnv]*2)

    vec_env.reset()
    terminal = False
    while not terminal:
        actions = [flatten_dictionary(StubEnv().action_space.sample())]*2
        actions = list(chain.from_iterable(actions))
        states, rewards, terminals, infos = vec_env.step(actions)
        assert len(states) == vec_env.num_envs
        terminal = np.any(terminals)


if __name__ == '__main__':
    test_optvecenv_step()
