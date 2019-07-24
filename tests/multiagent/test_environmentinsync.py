
from multiprocessing import Process
from threading import Thread

import gym
import pytest
import numpy as np

from custom_envs.multiagent import EnvironmentInSync
from custom_envs.utils.utils_venv import SubprocVecEnv

JOB_EXE = [Process, Thread]
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

def task_dummy_env(*sub_envs):
    try:
        dummy_env = SubprocVecEnv(sub_envs)
        dummy_env.step([sub.action_space.sample() for sub in sub_envs])
    except EOFError:
        print('Error!')
    finally:
        dummy_env.close()


def task_handle(env):
    env.handle_requests()
    env.close()

@pytest.mark.skip
def test_environmentinsync_vec_env():
    env_name = 'Optimize-v0'
    envs = [EnvironmentInSync(gym.make(env_name, data_set='iris'), 3)
            for _ in range(NUMBER_OF_PROCESSORS)]
    tasks = [Process(target=task_dummy_env,
                     args=[env.sub_envs[i] for env in envs])
             for i in range(3)]
    for task in tasks:
        task.start()
    tasks_envs = [Thread(target=task_handle, args=[env], daemon=True)
                  for env in envs]
    for task in tasks_envs:
        task.start()
    for task in tasks_envs:
        task.join()
    for task in tasks:
        task.join()


if __name__ == '__main__':
    test_environmentinsync_vec_env()
