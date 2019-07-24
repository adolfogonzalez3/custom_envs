'''Run an multi agent experiment.'''
import os
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from functools import partial

import gym
import pandas as pd
import tensorflow as tf
import stable_baselines.ddpg as ddpg
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines import PPO2, A2C, DDPG
from stable_baselines.common.misc_util import set_global_seeds

import custom_envs.utils.utils_file as utils_file
from custom_envs.multiagent.multienvserver import MultiEnvWrapper
from custom_envs.utils.utils_logging import Monitor
from custom_envs.utils.utils_venv import ThreadVecEnv
from custom_envs.envs.multioptlrs import MultiOptLRs
from custom_envs.envs.multioptimize import MultiOptimize
from custom_envs.vectorize.optvecenv import OptVecEnv

LOGGER = logging.getLogger(__name__)


def run_agent(envs, parameters):
    '''Train an agent.'''
    alg = parameters['alg']
    learning_rate = parameters['learning_rate']
    gamma = parameters['gamma']
    model_path = parameters['model_path']
    set_global_seeds(parameters.get('seed'))
    dummy_env = OptVecEnv(envs)
    if alg == 'PPO':
        model = PPO2(MlpPolicy, dummy_env, gamma=gamma,
                     learning_rate=learning_rate, verbose=1,
                     nminibatches=dummy_env.num_envs)
    elif alg == 'A2C':
        model = A2C(MlpPolicy, dummy_env, gamma=gamma,
                    learning_rate=learning_rate, verbose=1)
    else:
        model = DDPG(ddpg.MlpPolicy, dummy_env, gamma=gamma, verbose=1,
                     actor_lr=learning_rate/10, critic_lr=learning_rate)
    try:
        model.learn(total_timesteps=parameters.get('total_timesteps', 10**6))
    except tf.errors.InvalidArgumentError:
        LOGGER.error('Possible Nan, %s', str((alg, learning_rate, gamma)))
    finally:
        dummy_env.close()
        model.save(str(model_path))


def run_handle(env):
    '''Run handle until all experiments end.'''
    data = 0
    while data is not None:
        data = env.handle_requests()


def run_experiment(parameters):
    '''Set up and run an experiment.'''
    repository_path = Path(__file__).parent
    save_to = parameters['path']
    task_name = '{alg}-{learning_rate:.4f}-{gamma:.4f}-{seed:d}'
    task_name = task_name.format(**parameters)
    save_to = Path(save_to, task_name)
    with utils_file.create_directory(save_to, False, False) as log_dir:
        parameters['commit'] = utils_file.get_commit_hash(repository_path)
        utils_file.save_json(parameters, log_dir / 'hyperparams.json')
        parameters['model_path'] = str(log_dir / 'model.pkl')
        log_path = str(log_dir / 'monitor_{:d}')
        env_name = parameters['env_name']
        kwargs = parameters.setdefault('kwargs', {})
        env_callable = [
            partial(Monitor, gym.make(env_name, **kwargs),
                    log_path.format(i), allow_early_resets=True,
                    info_keywords=('loss', 'accuracy', 'actions_mean',
                                   'weights_mean', 'actions_std',
                                   'states_mean', 'grads_mean'),
                    chunk_size=parameters.setdefault('chunk_size', 5))
            for i in range(1)
        ]
        try:
            run_agent(env_callable, parameters)
        except RuntimeError as error:
            LOGGER.error('%s, %s', error, parameters)
    return parameters


def run_batch(commandline_args):
    '''Run a series of experiment(s) given a json file of descriptions.'''
    parser = argparse.ArgumentParser(prog=__file__+' batch')
    parser.add_argument("file_path", help="The path to the file with tasks.")
    args = parser.parse_args(commandline_args)
    dataframe = pd.read_json(args.file_path, orient='records', lines=True)
    parameters_list = dataframe.to_dict(orient='records')
    for parameters in parameters_list:
        if isinstance(parameters.get('kwargs'), str):
            parameters['kwargs'] = json.loads(parameters['kwargs'])
        print(parameters)
        run_experiment(parameters)


def run_test1(commandline_args):
    '''Test run the code.'''
    parameters = {
        "alg": "PPO", "env_name": "MultiOptimize-v0", "gamma": 0.9,
        "learning_rate": 0.01,
        "path": "results_optimize_LC_mnist",
        "total_timesteps": 5*10**7, "chunk_size": 10, "seed": 0,
        'kwargs': {'data_set': 'iris', 'batch_size': 32,
                   'max_batches': 40, 'version': 1,
                   'max_history': 25, "observation_version": 3,
                   'reward_version': 1}
    }
    print(parameters)
    run_experiment(parameters)


def run_test(commandline_args):
    '''Test run the code.'''
    parameters = {
        "alg": "PPO", "env_name": "MultiOptLRs-v0", "gamma": 0.9,
        "learning_rate": 0.01, "chunk_size": 10, "seed": 0,
        "path": "results_optlrs_LC_mnist",
        "total_timesteps": 5*10**7,
        'kwargs': {'data_set': 'mnist', 'c': 128,
                   'max_batches': 100, 'version': 1,
                   'max_history': 25, "observation_version": 3}
    }
    print(parameters)
    run_experiment(parameters)


def run_task(commandline_args):
    '''Run only one experiment in a json file.'''
    parser = argparse.ArgumentParser(prog=__file__+' run')
    parser.add_argument("path", help="The path to save the results.")
    parser.add_argument("--alg", help="The algorithm to use.",
                        default='PPO')
    parser.add_argument("--learning_rate", help="The learning rate to use.",
                        type=float)
    parser.add_argument("--gamma", help="The gamma to use.",
                        type=float)
    parser.add_argument("--seed", help="The seed to use.",
                        type=float)
    parser.add_argument("--env_name", help="The gamma to use.")
    parameters = vars(parser.parse_args(commandline_args))
    run_experiment(parameters)


def main(commandline_args):
    '''Main script function.'''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="The command to run.",
                        choices=['batch', 'run', 'test'])
    args = parser.parse_args(commandline_args[:1])
    tf.logging.set_verbosity(tf.logging.ERROR)
    if args.command == 'batch':
        run_batch(commandline_args[1:])
    elif args.command == 'run':
        run_task(commandline_args[1:])
    elif args.command == 'test':
        run_test(commandline_args[1:])


if __name__ == '__main__':

    # single_task_csv()
    # single_task_json()
    # loop_over_json_file()
    main(sys.argv[1:])
    # run_test()
