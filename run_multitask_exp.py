

import shutil
import sqlite3
import logging
from threading import Thread
import multiprocessing as mp
import concurrent.futures as confuture


from pathlib import Path
from tempfile import TemporaryDirectory


import tensorflow as tf
import stable_baselines.ddpg as ddpg
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C, DDPG
from stable_baselines.common.misc_util import set_global_seeds

from custom_envs.multiagent import EnvironmentInSync
from custom_envs.utils.utils_common import create_env
from custom_envs.utils.utils_venv import SubprocVecEnv

LOGGER = logging.getLogger(__name__)

def run_agent(envs, alg, learning_rate, gamma, seed, path):
    set_global_seeds(seed)
    # The algorithms require a vectorized environment to run
    #dummy_env = DummyVecEnv(envs)
    dummy_env = SubprocVecEnv(envs)
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
        model.save(path)
    except tf.errors.InvalidArgumentError:
        LOGGER.error('Possible Nan, {!s}'.format((alg, learning_rate, gamma)))
    finally:
        dummy_env.close()
        
def run_handle(env):
    data = 0
    while data is not None:
        data = env.handle_requests()

def run_experiment_multiagent(parameters):
    alg, learning_rate, gamma, seed, save_to = parameters
    env_name = 'Optimize-v0'
    num_of_envs = 1 if alg == 'DDPG' else 32
    task_name = '{}-{:.4f}-{:.4f}-{:d}'.format(alg, learning_rate, gamma, seed)
    save_to = Path(save_to, task_name)
    num_of_agents = 10
    with TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir, task_name)
        log_dir.mkdir(parents=True)
        with (log_dir / 'hyperparams.txt').open('wt') as json_file:
            json_file.write(str(parameters))
        envs = create_env(env_name, log_dir, num_of_envs, data_set='mnist')
        for i, env in enumerate(envs):
            env.seed(seed + i)
        save_paths = [str(log_dir / 'model_{:d}'.format(i))
                      for i in range(num_of_agents)]
        envs = [EnvironmentInSync(env, num_of_agents) for env in envs]
        sub_envs = [[env.sub_envs[i] for env in envs]
                    for i in range(num_of_agents)]
        task_args = []
        common = [alg, learning_rate, gamma, seed]
        for sub_env, save_path in zip(sub_envs, save_paths):
            task_args.append([sub_env] + common + [save_path])
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
        except RuntimeError as error:
            LOGGER.error('%s, %s', error, parameters)
            for task in tasks:
                task.terminate()
        finally:
            for task in tasks:
                task.join()
            shutil.make_archive(str(save_to), 'zip', str(log_dir))
    return parameters[:-1]

def main_database(filename_db, filepath_exp):
    with sqlite3.connect(filename_db) as conn:
        tasks_desc = conn.execute(('select alg, learning_rate, gamma, seed from '
                                   'hyperparameters where done = 0')).fetchall()
        tasks_desc = sorted(tasks_desc, key=lambda x: tuple(x[-1:0:-1]))
        tasks_desc = [t + (Path(filepath_exp).resolve(),)
                      for t in tasks_desc]
    for task in tasks_desc[:1]:
        args = run_experiment_multiagent(task)
        with sqlite3.connect(filename_db) as conn:
            with conn:
                conn.execute(('update hyperparameters set done = 1 '
                                'where alg=? and learning_rate=? and gamma=? '
                                'and seed=?'), args)

def main():
    filename_db = 'task_mnist_multi_agent.db'
    filepath_exp = './results_multiagent_mnist'
    with sqlite3.connect(filename_db) as conn:
        with conn:
            conn.execute(('update hyperparameters set done=2 where '
                          'alg=? and done=0'), ['DDPG'])
    main_database(filename_db, filepath_exp)

if __name__ == '__main__':
    main()
