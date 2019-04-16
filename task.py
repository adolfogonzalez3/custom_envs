

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

LOGGER = logging.getLogger(__name__)

def run_agent(*args):
    env, alg, learning_rate, gamma, seed, path = args

    set_global_seeds(seed)
    # The algorithms require a vectorized environment to run
    dummy_env = DummyVecEnv([lambda: env])
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
        LOGGER.error('Possible Nan, {!s}'.format(args[1:-1]))
    finally:
        model.save(path)
        env.close()
        

def run_experiment(args):
    alg, learning_rate, gamma, seed, save_to = args
    print('Starting: ', args[:-1])
    begin = time()
    env_name = 'Optimize-v0'
    os.makedirs(save_to, exist_ok=True)
    task_name = '{}-{:.4f}-{:.4f}-{:d}'.format(alg, learning_rate, gamma, seed)
    save_to = Path(save_to, env_name, task_name)
    N = 3
    with TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir, task_name)
        os.makedirs(log_dir)
        save_paths = [str(log_dir / 'model_{:d}'.format(i)) for i in range(N)]
        env = gym.make(env_name)
        env.seed(seed)
        env = Monitor(env, str(log_dir), allow_early_resets=True,
                      info_keywords=('objective', 'accuracy'))
        env = EnvironmentInSync(env, N)
        task_args = zip(env.sub_envs, [alg]*N, [learning_rate]*N,
                            [gamma]*N, [seed]*N, save_paths)
        tasks = [mp.Process(target=run_agent, args=targs)
                 for targs in task_args]
        for task in tasks:
            task.start()
        data = 0
        try:
            while data is not None:
                data = env.handle_requests()
            for task in tasks:
                task.join()
            env.close()
            if log_dir.is_dir():
                shutil.make_archive(str(save_to), 'zip', str(log_dir))
        except RuntimeError as error:
            LOGGER.error('{!s}, {!s}'.format(error, args[:-1]))
            for task in tasks:
                task.terminate()
            env.close()
    return args[:-1], time() - begin

def main_database():
    with sqlite3.connect('tests_1.db') as conn:
        tasks_desc = conn.execute(('select alg, lr, gamma, seed from '
                                   'experiments where done = 0')).fetchall()
        tasks_desc = sorted(tasks_desc, key=lambda x: x[-1])
        tasks_desc = [t + (Path('./results_multiagent').resolve(),)
                      for t in tasks_desc]

        with confuture.ProcessPoolExecutor(2) as executor:
            tasks = executor.map(run_experiment, tasks_desc[:10])
            for args, time_elapsed in tasks:
                with conn:
                    conn.execute(('update experiments set done = 1 '
                                  'where alg=? and lr=? and gamma=? '
                                  'and seed=?'),
                                 args)
                print('Finished: ', args, ' in ', time_elapsed)

def create_database():
    ALGS = ['A2C', 'DDPG', 'PPO']
    LEARNING_RATES = 10**np.linspace(-1, -3, 10)
    GAMMA = 10**np.linspace(0, -1, 10)
    SEED = list(range(20))
    conn = sqlite3.connect('tests_1.db')
    with conn:
        conn.execute(('create table experiments (alg text, lr real, '
                      'gamma real, seed integer, done integer)'))
        
        conn.executemany('insert into experiments values (?,?,?,?,?)',
                         product(ALGS, LEARNING_RATES, GAMMA, SEED, [0]))
    conn.close()

if __name__ == '__main__':
    if not Path('tests_1.db').is_file():
        create_database()
    main()
