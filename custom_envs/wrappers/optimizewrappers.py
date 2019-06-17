'''A module containing wrappers for the Optimize problems.'''
import numpy as np
from gym.core import Wrapper
from gym.spaces import Box, Dict

from custom_envs.utils.utils_common import History


class HistoryWrapper(Wrapper):
    '''
    Add past history to observation space.
    '''

    def __init__(self, env, max_history=5):
        named_shapes = {key: space.shape
                        for key, space in env.observation_space.spaces.items()}
        env.observation_space = Dict({
            key: Box(low=np.array([space.low]*max_history),
                     high=np.array([space.high]*max_history),
                     dtype=space.dtype)
            for key, space in env.observation_space.spaces.items()
        })
        self.history = History(max_history, **named_shapes)
        super().__init__(env)

    def step(self, action):
        state, reward, terminal, info = self.env.step(action)
        self.history.append(**state)
        return dict(self.history), reward, terminal, info

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self.history.reset(**state)
        return dict(self.history)

    def __repr__(self):
        string = '<{}{!r}{!r}>'
        return string.format(type(self).__name__, self.history, self.env)


class SubSetWrapper(Wrapper):
    '''
    Retrieve a subset of the total observations.
    '''

    def __init__(self, env, subset):
        '''
        :param env: (gym.Env) An environment with OpenAI gym API.
        :param subset: ([str]) A list of keys to retrieve from the observation
                               space. (Environment observation space must be
                               gym.spaces.Dict)
        '''
        env.observation_space = Dict(
            {key: env.observation_space[key] for key in subset}
        )
        self.subset = subset
        super().__init__(env)

    def step(self, action):
        state, reward, terminal, info = self.env.step(action)
        state = {name: state[name] for name in self.subset}
        return state, reward, terminal, info

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        return {name: state[name] for name in self.subset}

    def __repr__(self):
        string = '<{}{!r}{!r}>'
        return string.format(type(self).__name__, self.subset, self.env)
