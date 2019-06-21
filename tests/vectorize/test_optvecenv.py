'''A module for testing utils_venv.'''
from functools import partial

import numpy as np

from custom_envs.vectorize.optvecenv import OptVecEnv, flatten_dictionary
from custom_envs.envs.multioptimize import MultiOptimize


def test_optvecenv_reset():
    '''A test for vec_env classes reset method.'''
    env = MultiOptimize(max_batches=10)
    env_fn = partial(lambda: env)
    vec_env = OptVecEnv([env_fn])
    states = vec_env.reset()
    assert len(states) == len(env.observation_space.spaces)


def test_optvecenv_step():
    '''A test for vec_env classes step method.'''
    env = MultiOptimize(max_batches=10)
    env_fn = partial(lambda: env)
    vec_env = OptVecEnv([env_fn])
    
    terminal = False
    while not terminal:
        actions = flatten_dictionary(env.action_space.sample())
        states, rewards, terminals, infos = vec_env.step(actions)
        assert len(states) == vec_env.num_envs
        assert len(rewards) == vec_env.num_envs
        assert len(terminals) == vec_env.num_envs
        assert len(infos) == vec_env.num_envs
        terminal = np.any(terminals)
