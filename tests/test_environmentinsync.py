
from multiprocessing import Process
from threading import Thread

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

import gym
import pytest
import numpy as np

from custom_envs.multiagent import EnvironmentInSync

EXECUTORS = (ThreadPoolExecutor, ProcessPoolExecutor)
JOB_EXE = [Process, Thread]

def task_dummy_env(*sub_envs):
    try:
        dummy_env = SubprocVecEnv(sub_envs)
        print('stepping')
        dummy_env.step_async([np.ones((4,)) for _ in range(5)])
        print('waiting')
        dummy_env.step_wait()
        print('Done')
        dummy_env.close()
    except EOFError:
        print('Closing...')

def task_handle(env):
    print(env)
    print('beginning')
    env.handle_requests()
    print('Handled')
    env.close()

def test_environmentinsync_vec_env():
    env_name = 'Optimize-v0'
    envs = [EnvironmentInSync(gym.make(env_name), 3) for _ in range(5)]
    print('starting')
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
    print('Handled ALL')
    for task in tasks:
        task.join()



if __name__ == '__main__':
    test_environmentinsync_vec_env()
