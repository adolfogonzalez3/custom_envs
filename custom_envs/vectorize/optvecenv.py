'''Module that contains a class for vectorizing optimize environments.'''
from itertools import chain

import numpy as np
from stable_baselines.common.vec_env import VecEnv


def flatten_dictionary(dictionary):
    '''Flatten dictionary from the Optimize type environments.'''
    names_values = [(name, value) for name, value in dictionary.items()]
    _, values = list(zip(*sorted(names_values, key=lambda x: x[0])))
    return values


class OptEnvRunner:
    '''A class for running an Optimize type environment.'''
    def __init__(self, environment_fn):
        environment = environment_fn()
        self.names = sorted(iter(environment.action_space.spaces),
                            key=lambda x: x[0])
        name = self.names[0]
        self.environment = environment
        self.observation_space = environment.observation_space.spaces[name]
        self.action_space = environment.action_space.spaces[name]
        self.agents = len(environment.observation_space.spaces)
        for space in environment.observation_space.spaces.values():
            assert space == self.observation_space
        for space in environment.action_space.spaces.values():
            assert space == self.action_space

    def reset(self):
        '''Reset then environment and return the states for each parameter.'''
        return flatten_dictionary(self.environment.reset())

    def step(self, actions):
        '''Apply an action to the environment.'''
        actions = {name: action for name, action in zip(self.names, actions)}
        states, rewards, terminals, infos = self.environment.step(actions)
        states = flatten_dictionary(states)
        rewards = [rewards]*len(states)
        terminals = [terminals]*len(states)
        infos = [infos]*len(states)
        return states, rewards, terminals, infos


class OptVecEnv(VecEnv):
    '''A class for running multiple Optimize type environments.'''
    def __init__(self, environment_fns):
        '''Create all environments and ensure that all have similar space.'''
        self.environments = [OptEnvRunner(env) for env in environment_fns]
        observation_space = self.environments[0].observation_space
        action_space = self.environments[0].action_space
        agents = sum(env.agents for env in self.environments)
        super().__init__(agents, observation_space, action_space)
        self.states = []
        self.rewards = []
        self.terminals = []
        self.infos = []

    def reset(self):
        states = [env.reset() for env in self.environments]
        return np.stack(list(chain.from_iterable(states)))

    def step_async(self, actions):
        start = 0
        for environment in self.environments:
            actions_env = actions[start:(start + environment.agents)]
            states, rewards, terminals, infos = environment.step(actions_env)
            if np.all(terminals):
                states = environment.reset()
            self.states.append(states)
            self.rewards.append(rewards)
            self.terminals.append(terminals)
            self.infos.append(infos)
            start = start + environment.agents

    def step_wait(self):
        states = np.stack(list(chain.from_iterable(self.states)))
        rewards = np.stack(list(chain.from_iterable(self.rewards)))
        terminals = np.stack(list(chain.from_iterable(self.terminals)))
        infos = list(chain.from_iterable(self.infos))
        self.states = []
        self.rewards = []
        self.terminals = []
        self.infos = []
        return states, rewards, terminals, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        pass

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None,
                   **method_kwargs):
        pass
