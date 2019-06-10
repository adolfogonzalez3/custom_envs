
from functools import partial

import pytest
import gym
import numpy as np

from custom_envs.multiagent import EnvironmentInSync
from custom_envs.utils.utils_venv import SubprocVecEnv, ThreadVecEnv

VECENV_CLASSES = [SubprocVecEnv, ThreadVecEnv]
NUMBER_OF_PROCESSORS = 2


@pytest.mark.parametrize("vecenv_class", VECENV_CLASSES)
def test_vec_env_reset(vecenv_class):
    env_name = 'Optimize-v0'
    envs = [partial(gym.make, env_name, data_set='iris')
            for _ in range(NUMBER_OF_PROCESSORS)]
    vec_env = vecenv_class(envs)
    states = vec_env.reset()
    assert len(states) == NUMBER_OF_PROCESSORS

@pytest.mark.parametrize("vecenv_class", VECENV_CLASSES)
def test_vec_env_step(vecenv_class):
    env_name = 'Optimize-v0'
    envs = [partial(gym.make, env_name, data_set='iris')
            for _ in range(NUMBER_OF_PROCESSORS)]
    vec_env = vecenv_class(envs)
    terminal = False
    while not terminal:
        actions = [np.ones((12,)) for _ in range(NUMBER_OF_PROCESSORS)]
        states, rewards, terminal, info = vec_env.step(actions)
        assert len(states) == NUMBER_OF_PROCESSORS
        assert len(rewards) == NUMBER_OF_PROCESSORS
        assert len(terminal) == NUMBER_OF_PROCESSORS
        assert len(info) == NUMBER_OF_PROCESSORS
        terminal = np.all(terminal)
    

@pytest.mark.parametrize("vecenv_class", VECENV_CLASSES)
def test_vec_env_step(vecenv_class):
    env_name = 'Optimize-v0'
    envs = [partial(gym.make, env_name, data_set='iris')
            for _ in range(NUMBER_OF_PROCESSORS)]
    vec_env = vecenv_class(envs)
    terminal = False
    while not terminal:
        actions = [np.ones((12,)) for _ in range(NUMBER_OF_PROCESSORS)]
        states, rewards, terminal, info = vec_env.step(actions)
        assert len(states) == NUMBER_OF_PROCESSORS
        assert len(rewards) == NUMBER_OF_PROCESSORS
        assert len(terminal) == NUMBER_OF_PROCESSORS
        assert len(info) == NUMBER_OF_PROCESSORS
        terminal = np.all(terminal)
