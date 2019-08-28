'''Run an multi agent experiment.'''
import os
import logging
import argparse
from pathlib import Path
from functools import partial

import gym
import optuna
import tensorflow as tf
from tqdm import trange
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines import PPO2, A2C
from stable_baselines.common.misc_util import set_global_seeds

import custom_envs.utils.utils_file as utils_file
from custom_envs.utils.utils_logging import Monitor
from custom_envs.vectorize.optvecenv import OptVecEnv

LOGGER = logging.getLogger(__name__)


def run_agent(envs, parameters, trial):
    '''Train an agent.'''
    path = Path(parameters['path'])
    dummy_env = OptVecEnv(envs)
    set_global_seeds(parameters.setdefault('seed'))
    if parameters['alg'] == 'PPO':
        model = PPO2(
            MlpLstmPolicy, dummy_env, gamma=parameters['gamma'],
            learning_rate=parameters['learning_rate'], verbose=0,
            nminibatches=2
        )
    elif parameters['alg'] == 'A2C':
        model = A2C(
            MlpPolicy, dummy_env, gamma=parameters['gamma'],
            learning_rate=parameters['learning_rate'], verbose=0
        )
    try:
        timesteps = parameters['total_timesteps'] * dummy_env.agent_no_list[0]
        number_of_updates = parameters['total_timesteps'] // 128
        with trange(number_of_updates, leave=True) as progress:
            progress = iter(progress)

            def callback(local_vars, global_vars):
                if next(progress) % 100:
                    callback_env = local_vars['self'].env
                    metric = sum(
                        sum(r) for r in
                        callback_env.env_method('get_episode_rewards')
                    )
                    trial.report(metric, local_vars['update'])
                    #if trial.should_prune():
                    #    raise optuna.structs.TrialPruned()

            model.learn(total_timesteps=timesteps, callback=callback)
        return sum(sum(r) for r in dummy_env.env_method('get_episode_rewards'))
    finally:
        dummy_env.close()
        model.save(str(path / 'model.pkl'))


def get_parameters(trial):
    parameters = {
        'gamma': float(trial.suggest_loguniform(
            'gamma', 0.1, 0.99
        )),
        'learning_rate': float(trial.suggest_loguniform(
            'learning_rate', 1e-6, 1e-3
        )),
        'kwargs': {
            # 6 points to search
            # 'version': int(trial.suggest_int('version', 0, 5)),
            # 4 points to search
            # 'reward_version': 0,  # int(trial.suggest_int('reward_version', 0, 3)),
            # 2 points to search
            # 'action_version': int(trial.suggest_int('action_version', 0, 1)),
            # 5 points to search
            #'batch_size': int(trial.suggest_categorical(
            #    'batch_size', [2**i for i in range(7, 12)]
            #)),  # 4 points to search
            # 'observation_version': int(trial.suggest_int(
            #    'observation_version', 0, 3
            # )),
            # 5 points to search
            'max_history': int(trial.suggest_discrete_uniform(
                'max_history', 5, 25, 5
            )),
            # 5 points to search
            #'max_batches': int(trial.suggest_discrete_uniform(
            #    'max_batches', 100, 500, 100
            #))
            'max_batches': 500
        }
    }
    return parameters


def run_experiment(parameters, trial):
    '''Set up and run an experiment.'''
    parameters = parameters.copy()
    parameters.update(get_parameters(trial))
    path = Path(parameters['path']) / str(trial.number)
    path.mkdir()
    parameters['path'] = str(path)
    parameters['commit'] = utils_file.get_commit_hash(Path(__file__).parent)
    log_path = str(path / 'monitor_{:d}')
    kwargs = parameters.setdefault('kwargs', {})
    utils_file.save_json(parameters, path / 'parameters.json')
    wrapped_envs = [
        partial(
            Monitor,
            partial(
                gym.make, parameters['env_name'], **kwargs
            ),
            log_path.format(i),
            info_keywords=(
                'loss', 'actions_mean',
                'weights_mean', 'actions_std',
                'states_mean', 'grads_mean'
            ), chunk_size=parameters.setdefault('chunk_size', 5)
        ) for i in range(1)
    ]
    return -run_agent(wrapped_envs, parameters, trial)


def main():
    '''Main script function.'''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="The path to save the results.")
    parser.add_argument("--alg", help="The algorithm to use.", default='PPO')
    parser.add_argument("--env_name", help="The gamma to use.",
                        default='MultiOptLRs-v0')
    parser.add_argument('--total_timesteps', default=int(1e5), type=int,
                        help="Number of timesteps per training session")
    parser.add_argument('--data_set', help="The data set to use.",
                        default='iris')
    parser.add_argument('--trials', help="The number of trials to run.",
                        default=10, type=int)
    args = parser.parse_args()
    parameters = vars(args).copy()
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
    study = optuna.create_study(
        study_name=str(path.name),
        storage=storage, load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=args.trials)



if __name__ == '__main__':
    main()
