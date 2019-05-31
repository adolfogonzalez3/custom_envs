
import json
import shutil
import logging

from pathlib import Path
from functools import partial
from tempfile import TemporaryDirectory
from concurrent.futures import ProcessPoolExecutor

import gym
import pandas as pd
import tensorflow as tf
from stable_baselines import ddpg
from stable_baselines import PPO2, A2C, DDPG
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.misc_util import set_global_seeds

from custom_envs.utils.utils_logging import Monitor
from custom_envs.utils.utils_venv import SubprocVecEnv

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
        model.learn(total_timesteps=10**4)
    except tf.errors.InvalidArgumentError:
        LOGGER.error('Possible Nan, {!s}'.format((alg, learning_rate, gamma)))
    finally:
        dummy_env.close()
        model.save(str(path))
        


def run_experiment(parameters):
    env_name = parameters['env_name']
    alg = parameters['alg']
    learning_rate = parameters['learning_rate']
    gamma = parameters['gamma']
    seed = parameters['seed']
    save_to = parameters['path']
    num_of_envs = 1 if alg == 'DDPG' else parameters.get('num_of_envs', 32)
    task_name = '{}-{:.4f}-{:.4f}-{:d}'.format(alg, learning_rate, gamma, seed)
    save_to = Path(save_to, task_name).resolve()
    with TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir, task_name)
        log_dir.mkdir(parents=True)
        with (log_dir / 'hyperparams.json').open('wt') as json_file:
            json.dump(parameters, json_file)
        model_path = str(log_dir / 'model.pkl')
        log_path = str(log_dir / 'monitor_{:d}')
        envs_callable = [partial(Monitor, gym.make(env_name),
                                 log_path.format(i),
                                 allow_early_resets=True,
                                 info_keywords=('objective', 'accuracy'),
                                 chunk_size=1024)
                         for i in range(num_of_envs)]
        try:
            run_agent(envs_callable, alg, learning_rate, gamma, seed,
                      model_path)
        except RuntimeError as error:
            LOGGER.error('%s, %s', error, parameters)
        finally:
            shutil.make_archive(str(save_to), 'zip', str(log_dir))
    return parameters


def single_task_csv():
    import sys

    alg, lr, gamma, seed, env_name, save_to = sys.argv[1].rstrip().split(',')
    lr = float(lr)
    gamma = float(gamma)
    seed = int(seed)
    parameters = {'env_name': env_name, 'alg': alg, 'learning_rate': lr,
                  'gamma': gamma, 'seed': seed, 'path': save_to}
    run_experiment(parameters)

def loop_over_json_file():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="The path to the file with tasks.")
    args = parser.parse_args()
    dataframe = pd.read_json(args.file_path, orient='records', lines=True)
    parameters_list = dataframe.to_dict(orient='records')
    with ProcessPoolExecutor(8) as executor:
        executor.map(run_experiment, parameters_list)
        #for parameters in parameters_list:
        #    print(parameters)
        #    run_experiment(parameters)

def single_task_json():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="The path to the file with tasks.")
    parser.add_argument("line_no", help="The line number of load.",
                        type=int)
    args = parser.parse_args()
    assert args.line_no > 0
    with Path(args.file_path).resolve().open('rt') as json_file:
        for _ in range(args.line_no):
            json_line = json_file.readline()
    parameters = json.loads(json_line)
    run_experiment(parameters)

if __name__ == '__main__':
    # single_task_csv()
    # single_task_json()
    loop_over_json_file()
