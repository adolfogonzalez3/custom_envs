'''A module that contains an abstract environment class.'''

from abc import ABC, abstractmethod
from contextlib import contextmanager

import numpy.random as npr
from gym import Env
from gym.utils.seeding import np_random

from custom_envs.utils.utils_common import use_random_state


class BaseEnvironment(Env):
    '''
    An abstract class inherited by all environments in the package.
    '''

    def __init__(self):
        self.random_generator, _ = np_random()
        self.current_step = 0

    def seed(self, seed=None):
        '''
        Seed the environment's random generator.

        :param seed: (int or None) If an integer then seed is used to seed the
                                   random generator otherwise if None then
                                   use seed = 0.
        '''
        self.random_generator, _ = np_random(seed)

    def step(self, action):
        '''
        Take one step in the environment.

        :param action: (numpy.array) An action that follows the rules of the
                                     action space.
        '''
        with use_random_state(self.random_generator):
            state, reward, terminal, info = self._step(action)
        info['episode'] = {'r': reward, 'l': self.current_step}
        self.current_step += 1
        return state, reward, terminal, info

    def reset(self):
        '''
        Reset the environment.
        '''
        self.current_step = 0
        with use_random_state(self.random_generator):
            return self._reset()

    @abstractmethod
    def _step(self, action):
        pass

    @abstractmethod
    def _reset(self):
        pass