'''Run a single agent experiment.'''
import sys
import shutil
import logging
import argparse

from pathlib import Path
from functools import partial
from tempfile import TemporaryDirectory

import gym
import pandas as pd
import tensorflow as tf
from stable_baselines import ddpg
from stable_baselines import PPO2, A2C, DDPG
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.misc_util import set_global_seeds

import custom_envs.utils.utils_file as utils_file
from custom_envs.utils.utils_logging import Monitor
from custom_envs.utils.utils_venv import ThreadVecEnv

LOGGER = logging.getLogger(__name__)


def run_agent(envs, alg, learning_rate, gamma, seed, path):
    '''Train an agent.'''
    set_global_seeds(seed)
    dummy_env = ThreadVecEnv(envs)
    if alg == 'PPO':
        model = PPO2(MlpLstmPolicy, dummy_env, gamma=gamma,
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
        LOGGER.error('Possible Nan, %s', str((alg, learning_rate, gamma)))
    finally:
        dummy_env.close()
        model.save(str(path))


def run_experiment(parameters):
    '''Set up and run an experiment.'''
    repository_path = Path(__file__).parent
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
        parameters['num_of_envs'] = num_of_envs
        parameters['commit'] = utils_file.get_commit_hash(repository_path)
        utils_file.save_json(parameters, log_dir / 'hyperparams.json')
        model_path = str(log_dir / 'model.pkl')
        log_path = str(log_dir / 'monitor_{:d}')
        envs_callable = [partial(Monitor,
                                 gym.make(env_name, data_set='mnist',
                                          batch_size=32,
                                          version=3),
                                 log_path.format(i),
                                 allow_early_resets=True,
                                 info_keywords=('objective', 'accuracy'),
                                 chunk_size=1)
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
    '''Run an experiment given input in the command line.'''
    alg, lrate, gamma, seed, env_name, path = sys.argv[1].rstrip().split(',')
    lrate = float(lrate)
    gamma = float(gamma)
    seed = int(seed)
    parameters = {'env_name': env_name, 'alg': alg, 'learning_rate': lrate,
                  'gamma': gamma, 'seed': seed, 'path': path}
    run_experiment(parameters)


def loop_over_json_file():
    '''Run a series of experiment(s) given a json file of descriptions.'''
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="The path to the file with tasks.")
    args = parser.parse_args()
    dataframe = pd.read_json(args.file_path, orient='records', lines=True)
    parameters_list = dataframe.to_dict(orient='records')
    # with ProcessPoolExecutor(1) as executor:
    #    executor.map(run_experiment, parameters_list)
    for parameters in parameters_list[:1]:
        print(parameters)
        run_experiment(parameters)


def main():
    '''Test run the code.'''
    parameters = {"alg": "PPO", "env_name": "OptLRs-v0", "gamma": 0.9,
                  "learning_rate": 0.001, "path": "results_iris2", "seed": 0}
    run_experiment(parameters)


def single_task_json():
    '''Run only one experiment in a json file.'''
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="The path to the file with tasks.")
    parser.add_argument("line_no", help="The line number of load.",
                        type=int)
    args = parser.parse_args()
    assert args.line_no > 0
    parameters = utils_file.load_json(args.file_path, line_no=args.line_no)
    print(parameters)
    run_experiment(parameters)


if __name__ == '__main__':
    # single_task_csv()
    # single_task_json()
    # loop_over_json_file()
    main()
