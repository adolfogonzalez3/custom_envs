

import os
import shutil
import sqlite3
import logging
import multiprocessing as mp
import concurrent.futures as confuture

from time import time
from pathlib import Path
from itertools import product
from threading import Thread
from tempfile import TemporaryDirectory

import gym
import numpy as np

import tensorflow as tf
from stable_baselines.bench import Monitor
import stable_baselines.ddpg as ddpg
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2, A2C, DDPG
from stable_baselines.common.misc_util import set_global_seeds

from custom_envs.multiagent import EnvironmentInSync
from custom_envs.utils import create_env

LOGGER = logging.getLogger(__name__)

def run_agent(envs, alg, learning_rate, gamma, seed, path):
    set_global_seeds(seed)
    # The algorithms require a vectorized environment to run
    #dummy_env = DummyVecEnv([lambda: env])
    dummy_env = DummyVecEnv(envs)
    if alg == 'PPO':
        model = PPO2(MlpPolicy, dummy_env, gamma=gamma,
                     learning_rate=learning_rate, verbose=1)
    elif alg == 'A2C':
        model = A2C(MlpPolicy, dummy_env, gamma=gamma,
                    learning_rate=learning_rate, verbose=1)
    else:
        model = DDPG(ddpg.MlpPolicy, dummy_env, gamma=gamma, verbose=1,
                     actor_lr=learning_rate/10, critic_lr=learning_rate)
    
    try:
        model.learn(total_timesteps=10**6)
    # tf.errors.InvalidArgumentError as error
    except tf.errors.InvalidArgumentError as error:
        LOGGER.error('Possible Nan, {!s}'.format((alg, learning_rate, gamma)))
    finally:
        model.save(path)
        #env.close()
        for env in envs:
            env.close()
        
def run_handle(env):
    data = 0
    while data is not None:
        data = env.handle_requests()

def run_experiment_multiagent(args):
    alg, learning_rate, gamma, seed, save_to = args
    begin = time()
    env_name = 'Optimize-v0'
    num_of_envs = 1 if alg == 'DDPG' else 32
    task_name = '{}-{:.4f}-{:.4f}-{:d}'.format(alg, learning_rate, gamma, seed)
    save_to = Path(save_to, env_name, task_name)
    number_of_agents = 3
    with TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir, task_name)
        log_dir.mkdir(parents=True)
        envs = create_env(env_name, log_dir, num_of_envs)
        for i, env in enumerate(envs):
            env.seed(seed + i)
        save_paths = [str(log_dir / 'model_{:d}'.format(i))
                      for i in range(number_of_agents)]
        envs = [EnvironmentInSync(env, number_of_agents) for env in envs]
        sub_envs = [[env.sub_envs[i] for env in envs]
                    for i in range(number_of_agents)]
        task_args = zip(sub_envs, [alg]*number_of_agents,
                        [learning_rate]*number_of_agents,
                        [gamma]*number_of_agents, [seed]*number_of_agents,
                        save_paths)
        tasks = [mp.Process(target=run_agent, args=targs)
                 for targs in task_args]
        for task in tasks:
            task.start()
        try:
            tasks_env = [Thread(target=run_handle, args=[env])
                         for env in envs]
            for task in tasks_env:
                task.start()
            for task in tasks_env:
                task.join()
            shutil.make_archive(str(save_to), 'zip', str(log_dir))
        except RuntimeError as error:
            LOGGER.error('%s, %s', error, args[:-1])
            for task in tasks:
                task.terminate()
            for env in envs:
                env.close()
    return args[:-1], time() - begin

def main_database(filename_db, filepath_exp):
    with sqlite3.connect(filename_db) as conn:
        tasks_desc = conn.execute(('select alg, lr, gamma, seed from '
                                   'experiments where done = 0')).fetchall()
        tasks_desc = sorted(tasks_desc, key=lambda x: tuple(x[-1:0:-1]))
        tasks_desc = [t + (Path(filepath_exp).resolve(),)
                      for t in tasks_desc]
    with confuture.ProcessPoolExecutor(1) as executor:
        tasks = executor.map(run_experiment, tasks_desc[:2])
        for args, time_elapsed in tasks:
            with sqlite3.connect(filename_db) as conn:
                with conn:
                    conn.execute(('update experiments set done = 1 '
                                  'where alg=? and lr=? and gamma=? '
                                  'and seed=?'), args)
                print('Finished: ', args, ' in ', time_elapsed)

def create_database(filename_db):
    ALGS = ['A2C', 'DDPG', 'PPO']
    LEARNING_RATES = 10**np.linspace(-1, -3, 10)
    GAMMA = 10**np.linspace(0, -1, 10)
    SEED = list(range(20))
    with sqlite3.connect(filename_db) as conn:
        with conn:
            conn.execute(('create table experiments (alg text, lr real, '
                          'gamma real, seed integer, done integer)'))
            
            conn.executemany('insert into experiments values (?,?,?,?,?)',
                             product(ALGS, LEARNING_RATES, GAMMA, SEED, [0]))

if __name__ == '__main__':
    filename_db = 'tests_1.db'
    filepath_exp = './results_multiagent'
    if not Path(filename_db).is_file():
        create_database()
    with sqlite3.connect(filename_db) as conn:
        with conn:
            conn.execute(('update experiments set done=2 where '
                          'alg=? and done=0'), 'DDPG')
    main_database(filename_db, filepath_exp)
