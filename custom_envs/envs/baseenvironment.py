'''A module that contains an abstract environment class.'''

from abc import abstractmethod

from gym import Env
from gym.utils.seeding import np_random

from custom_envs.utils.utils_math import use_random_state


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
        self.current_step += 1
        with use_random_state(self.random_generator):
            state, reward, terminal, info = self.base_step(action)
        info['episode'] = {'r': reward, 'l': self.current_step}
        return state, reward, terminal, info

    def reset(self):
        '''
        Reset the environment.
        '''
        self.current_step = 0
        with use_random_state(self.random_generator):
            return self.base_reset()

    @abstractmethod
    def base_step(self, action):
        pass

    @abstractmethod
    def base_reset(self):
        pass


class BaseMultiEnvironment(BaseEnvironment):
    '''
    An abstract class inherited by all Multi agent environments in the package.
    '''
    AGENT_FMT = 'parameter-{:d}'
