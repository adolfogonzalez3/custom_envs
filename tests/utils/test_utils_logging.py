'''Tests utils_logging module.'''

from tempfile import TemporaryDirectory
from pathlib import Path

import pytest
import numpy as np
import numpy.random as npr
import pandas as pd
import gym

import custom_envs.utils.utils_logging as utils

TERMINAL_STEPS = 10
STEPS = npr.randint(TERMINAL_STEPS, 100, size=1)
ACTION_SIZE = 10


class StubEnv(gym.Env):
    '''Environment for testing.'''

    def __init__(self):
        self.counter = 0
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(0, np.max(STEPS),
                                                shape=(ACTION_SIZE,))

    def step(self, action):
        '''Stub.'''
        self.counter += 1
        return ([self.counter]*ACTION_SIZE, self.counter, self.terminal(),
                {'half': self.counter // 2})

    def terminal(self):
        '''Stub.'''
        return self.counter >= TERMINAL_STEPS

    def render(self, mode='human'):
        '''Stub.'''

    def reset(self):
        '''Stub.'''
        self.counter = 0
        return [0]*ACTION_SIZE


@pytest.fixture(scope="module")
def save_path():
    '''Yield a temporary directory.'''
    with TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope="module")
def env_obj():
    '''Return a stub environment.'''
    return StubEnv()


def test_monitor_init(save_path, env_obj):
    '''Tests __init__ method of monitor.'''
    monitor = utils.Monitor(env_obj, Path(save_path) / 'test_init')
    assert str(monitor.file_path).endswith(utils.Monitor.EXT)
    monitor = utils.Monitor(env_obj, None)
    assert monitor.file_path is None


@pytest.mark.parametrize("step", STEPS)
def test_monitor_save_without_info(save_path, env_obj, step):
    '''Tests monitor's save method.'''
    save_file = Path(save_path) / 'test_save_without_info'
    monitor = utils.Monitor(env_obj, save_file, chunk_size=step+1)
    monitor.reset()
    for _ in range(step):
        _, _, terminal, _ = monitor.step(None)
        if terminal:
            monitor.reset()
    assert not save_file.with_suffix(utils.Monitor.EXT).is_file()
    monitor.save()
    with save_file.with_suffix(utils.Monitor.EXT).open('rt') as csv:
        print('Printing: ', csv.read())
    assert save_file.with_suffix(utils.Monitor.EXT).is_file()
    dataframe = pd.read_csv(save_file.with_suffix(utils.Monitor.EXT))
    assert len(dataframe) == (step // TERMINAL_STEPS)
    assert 'l' in dataframe.columns
    assert 'r' in dataframe.columns
    assert 't' in dataframe.columns
    assert 'half' not in dataframe.columns


@pytest.mark.parametrize("step", STEPS)
def test_monitor_save_with_info(save_path, env_obj, step):
    '''Tests monitor's save method.'''
    save_file = Path(save_path) / 'test_save_with_info'
    monitor = utils.Monitor(env_obj, save_file, chunk_size=step+1,
                            info_keywords=('half',))
    monitor.reset()
    for _ in range(step):
        _, _, terminal, _ = monitor.step(None)
        if terminal:
            monitor.reset()
    assert not save_file.with_suffix(utils.Monitor.EXT).is_file()
    monitor.save()
    with save_file.with_suffix(utils.Monitor.EXT).open('rt') as csv:
        print('Printing: ', csv.read())
    assert save_file.with_suffix(utils.Monitor.EXT).is_file()
    dataframe = pd.read_csv(save_file.with_suffix(utils.Monitor.EXT))
    assert len(dataframe) == (step // TERMINAL_STEPS)
    assert 'l' in dataframe.columns
    assert 'r' in dataframe.columns
    assert 't' in dataframe.columns
    assert 'half' in dataframe.columns


def test_monitor_reset(save_path, env_obj):
    '''Tests monitor's reset method.'''
    save_file = Path(save_path) / 'test_reset'
    monitor = utils.Monitor(env_obj, save_file)
    with pytest.raises(RuntimeError):
        monitor.step(None)
    monitor.reset()
    with pytest.raises(RuntimeError):
        monitor.reset()


def test_monitor_step(save_path, env_obj):
    '''Test monitor's step method.'''
    save_file = Path(save_path) / 'test_step'
    monitor = utils.Monitor(env_obj, save_file, chunk_size=1,
                            info_keywords=('half',))
    monitor.reset()
    for _ in range(TERMINAL_STEPS-1):
        monitor.step(None)
    assert not save_file.with_suffix(utils.Monitor.EXT).is_file()
    state, reward, terminal, info = monitor.step(None)
    assert len(state) == TERMINAL_STEPS
    assert state[0] == TERMINAL_STEPS
    assert reward == TERMINAL_STEPS
    assert terminal
    assert 'half' in info
    assert save_file.with_suffix(utils.Monitor.EXT).is_file()
    with pytest.raises(RuntimeError):
        monitor.step(None)
