
from multiprocessing import Process
from threading import Thread

import gym
import pytest
import numpy as np

from custom_envs.multiagent import EnvironmentInSync
from custom_envs.utils.utils_venv import SubprocVecEnv

JOB_EXE = [Process, Thread]
NUMBER_OF_PROCESSORS = 2


def task_dummy_env(*sub_envs):
    try:
        dummy_env = SubprocVecEnv(sub_envs)
        dummy_env.step([np.ones((4,)) for _ in range(NUMBER_OF_PROCESSORS)])
    except EOFError:
        print('Error!')
    finally:
        dummy_env.close()


def task_handle(env):
    env.handle_requests()
    env.close()


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
