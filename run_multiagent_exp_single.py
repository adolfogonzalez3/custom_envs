'''Run an multi agent experiment.'''
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from threading import Thread
from functools import partial
from tempfile import TemporaryDirectory

import pandas as pd
import tensorflow as tf
import stable_baselines.ddpg as ddpg
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, A2C, DDPG
from stable_baselines.common.misc_util import set_global_seeds

import custom_envs.utils.utils_file as utils_file
from custom_envs.multiagent import MultiEnvServer
from custom_envs.utils.utils_logging import Monitor
from custom_envs.utils.utils_venv import ThreadVecEnv
from custom_envs.envs.multioptlrs import MultiOptLRs
from custom_envs.envs.multioptimize import MultiOptimize

LOGGER = logging.getLogger(__name__)


def run_agent(envs, parameters):
    '''Train an agent.'''
    alg = parameters['alg']
    learning_rate = parameters['learning_rate']
    gamma = parameters['gamma']
    model_path = parameters['model_path']
    set_global_seeds(parameters.get('seed'))
    dummy_env = ThreadVecEnv(envs)
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
    with TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir, task_name)
        log_dir.mkdir(parents=True)
        parameters['commit'] = utils_file.get_commit_hash(repository_path)
        utils_file.save_json(parameters, log_dir / 'hyperparams.json')
        env = MultiOptimize(**parameters.get('kwargs', {}))
        env.seed(parameters.get('seed'))
        parameters['model_path'] = str(log_dir / 'model.pkl')
        log_path = str(log_dir / 'monitor_{:d}')
        main_environment = MultiEnvServer(env)
        env_callable = [partial(Monitor, subenv, log_path.format(i),
                                allow_early_resets=True,
                                info_keywords=('objective', 'accuracy'),
                                chunk_size=10)
                        for i, subenv in
                        enumerate(main_environment.sub_environments.values())]
        try:
            task = Thread(target=run_handle, args=[main_environment],
                          daemon=True)
            taskrun = Thread(target=run_agent, args=[env_callable, parameters])
            task.start()
            taskrun.start()
            taskrun.join()
            task.join()
        except RuntimeError as error:
            LOGGER.error('%s, %s', error, parameters)
        finally:
            shutil.make_archive(str(save_to), 'zip', str(log_dir))
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


def run_test(commandline_args):
    '''Test run the code.'''
    parameters = {"alg": "PPO", "env_name": "MultiOptimize-v0", "gamma": 0.5,
                  "learning_rate": 0.001, "path": "results_iriss", "seed": 0,
                  "total_timesteps": 10**6,
                  'kwargs': {'data_set': 'iris', 'batch_size': 32,
                             'max_batches': 40}}
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
