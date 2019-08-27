'''Module that contains a class for vectorizing optimize environments.'''
from itertools import chain
from functools import partial

import numpy as np

from custom_envs.vectorize.concurrentvecenv import ThreadVecEnv


def flatten_dictionary(dictionary):
    '''Flatten dictionary from the Optimize type environments.'''
    names_values = [(name, value) for name, value in dictionary.items()]
    _, values = zip(*sorted(names_values, key=lambda x: x[0]))
    return values


class OptEnvRunner:
    '''A class for running an Optimize type environment.'''

    def __init__(self, environment_fn):
        environment = environment_fn()
        self._names = sorted(iter(environment.action_space.spaces),
                             key=lambda x: x[0])
        name = self._names[0]
        self._environment = environment
        self.observation_space = environment.observation_space.spaces[name]
        self.action_space = environment.action_space.spaces[name]
        self._num_agents = len(environment.observation_space.spaces)
        for space in environment.observation_space.spaces.values():
            assert space == self.observation_space
        for space in environment.action_space.spaces.values():
            assert space == self.action_space

    def reset(self):
        '''Reset then environment and return the states for each parameter.'''
        return flatten_dictionary(self._environment.reset())

    def step(self, actions):
        '''Apply an action to the environment.'''
        actions = {name: action for name, action in zip(self._names, actions)}
        states, reward, terminal, info = self._environment.step(actions)
        states = flatten_dictionary(states)
        reward = [reward]*self._num_agents
        terminal = [terminal]*self._num_agents
        info = [info]*self._num_agents
        return states, reward, terminal, info

    def close(self):
        self._environment.close()

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._environment, attr)


class OptVecEnv(ThreadVecEnv):
    '''A class for running multiple Optimize type environments.'''

    def __init__(self, environment_fns, callbacks=()):
        '''Create all environments and ensure that all have similar space.'''
        environment_fns = [
            partial(OptEnvRunner, env) for env in environment_fns
        ]
        super().__init__(environment_fns)
        self.agent_no_list = self.get_attr('_num_agents')
        self.num_envs = sum(self.agent_no_list)
        self.callbacks = callbacks

    def step_async(self, actions):
        grouped_actions = []
        start = 0
        for agent_no in self.agent_no_list:
            grouped_actions.append(actions[start:(start + agent_no)])
            start = start + agent_no
        super().step_async(grouped_actions)

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        states = np.stack(list(chain.from_iterable(obs)))
        rewards = np.stack(list(chain.from_iterable(rews)))
        terminals = np.stack(list(chain.from_iterable(dones)))
        infos = list(chain.from_iterable(infos))
        for callback in self.callbacks:
            callback(states, rewards, terminals, infos)
        return states, rewards, terminals, infos

    def reset(self):
        return np.stack(list(chain.from_iterable(super().reset())))
