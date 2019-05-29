
import json
import shutil
import logging
import sqlite3
import concurrent.futures as confuture

from time import time
from pathlib import Path
from functools import partial
from itertools import product
from tempfile import TemporaryDirectory

import tensorflow as tf
from stable_baselines import ddpg
from stable_baselines import PPO2, A2C, DDPG
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.misc_util import set_global_seeds

from custom_envs.utils.utils_logging import Monitor
from custom_envs.utils.utils_venv import SubprocVecEnv
from custom_envs.envs.optimize import Optimize

ENV_NAMES = ('Optimize-v0', 'OptLR-v0', 'OptLRs-v0')
LOGGER = logging.getLogger(__name__)


def run_agent(envs, alg, learning_rate, gamma, seed, path):
    set_global_seeds(seed)
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
    except tf.errors.InvalidArgumentError:
        LOGGER.error('Possible Nan, {!s}'.format((alg, learning_rate, gamma)))
    finally:
        dummy_env.close()
        model.save(str(path))
        


def run_experiment(env_name, alg, learning_rate, gamma, seed, save_to):
    expr_id = (env_name, alg, learning_rate, gamma, seed)
    num_of_envs = 1 if alg == 'DDPG' else 2
    task_name = '{}-{:.4f}-{:.4f}-{:d}'.format(alg, learning_rate, gamma, seed)
    save_to = Path(save_to, env_name, task_name).resolve()
    with TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir, task_name)
        log_dir.mkdir(parents=True)
        with (log_dir / 'hyperparams.json').open('wt') as js_file:
            json.dump({'env_name': env_name, 'alg': alg, 'gamma': gamma,
                       'learning_rate': learning_rate, 'seed': seed}, js_file)
        model_path = str(log_dir / 'model.pkl')
        log_path = str(log_dir / 'monitor_{:d}')
        #envs = create_env(env_name, log_dir, num_of_envs)
        #for i, env in enumerate(envs):
        #    env.seed(seed + i*num_of_envs)
        #envs_callable = [partial(lambda e: e, env) for env in envs]
        envs_callable = [partial(Monitor, Optimize(), log_path.format(i),
                                 allow_early_resets=True,
                                 info_keywords=('objective', 'accuracy'),
                                 chunk_size=1024)
                         for i in range(num_of_envs)]
        try:
            run_agent(envs_callable, alg, learning_rate, gamma, seed,
                      model_path)
            shutil.make_archive(str(save_to), 'zip', str(log_dir))
        except RuntimeError as error:
            LOGGER.error('%s, %s', error, expr_id)
        finally:
            #for env in envs:
            #    env.close()
            pass
    return expr_id


def main_single_task():
    import sys

    alg, lr, gamma, seed, env_name, save_to = sys.argv[1].rstrip().split(',')
    lr = float(lr)
    gamma = float(gamma)
    seed = int(seed)
    path = Path(save_to).resolve()
    run_experiment(env_name, alg, lr, gamma, seed, save_to)


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


if __name__ == '__main__':
    import argparse
    main_single_task()
    # create_tasks_file('tasks.csv')
    #parser = argparse.ArgumentParser()
    #parser.add_argument('type', choices=['csv', 'sqlite3'])

    #args = parser.parse_args()
