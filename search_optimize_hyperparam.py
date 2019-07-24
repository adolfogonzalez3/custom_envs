'''Run an multi agent experiment.'''
import os
import logging
import argparse
from pathlib import Path
from functools import partial

import gym
import optuna
import tensorflow as tf
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, A2C
from stable_baselines.common.misc_util import set_global_seeds

import custom_envs.utils.utils_file as utils_file
from custom_envs.utils.utils_logging import Monitor
from custom_envs.vectorize.optvecenv import OptVecEnv

LOGGER = logging.getLogger(__name__)


def run_agent(envs, parameters):
    '''Train an agent.'''
    path = Path(parameters['path'])
    dummy_env = OptVecEnv(envs)
    set_global_seeds(parameters.setdefault('seed'))
    if parameters['alg'] == 'PPO':
        model = PPO2(MlpPolicy, dummy_env, gamma=parameters['gamma'],
                     learning_rate=parameters['learning_rate'], verbose=0,
                     nminibatches=dummy_env.num_envs)
    elif parameters['alg'] == 'A2C':
        model = A2C(MlpPolicy, dummy_env, gamma=parameters['gamma'],
                    learning_rate=parameters['learning_rate'], verbose=0)
    try:
        timesteps = dummy_env.num_envs * parameters['total_timesteps']
        model.learn(total_timesteps=timesteps)
    finally:
        dummy_env.close()
        model.save(str(path / 'model.pkl'))


def optuna_callback(trial, episode_info):
    if episode_info['done']:
        trial.report(episode_info['info']['loss'], episode_info['episode'])
        if trial.should_prune():
            raise optuna.structs.TrialPruned()


def run_experiment(parameters, trial):
    '''Set up and run an experiment.'''
    parameters = parameters.copy()
    batch_size = [2**i for i in range(5, 12)]
    batch_size = int(trial.suggest_categorical('batch_size', batch_size))
    obs_version = int(trial.suggest_int('observation_version', 0, 3))
    max_history = int(trial.suggest_discrete_uniform('max_history', 5, 50, 5))
    learning_rate = float(trial.suggest_loguniform('learning_rate', 1e-5, 1e0))
    parameters.update({
        'gamma': float(trial.suggest_uniform('gamma', 0.1, 1.0)),
        'learning_rate': learning_rate
    })
    parameters.setdefault('kwargs', {}).update({
        'batch_size': batch_size,
        'version': int(trial.suggest_int('version', 0, 5)),
        'observation_version': obs_version, 'max_history': max_history,
        'reward_version': int(trial.suggest_int('reward_version', 0, 1)),
        'action_version': int(trial.suggest_int('action_version', 0, 1))
    })
    path = Path(parameters['path']) / str(trial.number)
    path.mkdir()
    parameters['path'] = str(path)
    parameters['commit'] = utils_file.get_commit_hash(Path(__file__).parent)
    log_path = str(path / 'monitor_{:d}')
    kwargs = parameters.setdefault('kwargs', {})
    utils_file.save_json(parameters, path / 'parameters.json')
    callback = partial(optuna_callback, trial)
    env = Monitor(
        gym.make(parameters['env_name'], **kwargs), log_path.format(0),
        info_keywords=(
            'loss', 'accuracy', 'actions_mean',
            'weights_mean', 'actions_std',
            'states_mean', 'grads_mean'
        ),
        chunk_size=parameters.setdefault('chunk_size', 5),
        callbacks=[callback]
    )
    run_agent([partial(lambda x: x, env)], parameters)
    return env.last_info['loss']


def main():
    '''Main script function.'''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="The path to save the results.")
    parser.add_argument("--alg", help="The algorithm to use.",
                        default='PPO')
    parser.add_argument("--env_name", help="The gamma to use.",
                        default='MultiOptLRs-v0')
    parser.add_argument('--total_timesteps', default=int(1e6), type=int,
                        help="Number of timesteps per training session")
    parser.add_argument('--data_set', help="The data set to use.",
                        default='mnist')
    parser.add_argument('--trials', help="The number of trials to run.",
                        default=10, type=int)
    args = parser.parse_args()
    parameters = vars(args)
    del parameters['trials']
    path = Path(parameters['path'])
    if not path.exists():
        path.mkdir()
        parameters['kwargs'] = {'data_set': parameters['data_set']}
        utils_file.save_json(parameters, path / 'parameters.json')
    else:
        if (path / 'study.db').exists():
            print('Directory exists. Using existing study and parameters.')
            parameters = utils_file.load_json(path / 'parameters.json')
        else:
            raise FileExistsError(('Directory already exists and is not a '
                                   'study.'))
    objective = partial(run_experiment, parameters)
    storage = 'sqlite:///' + str(path / 'study.db')
    study = optuna.create_study(study_name=str(path.name),
                                storage=storage, load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=args.trials)


if __name__ == '__main__':
    main()
