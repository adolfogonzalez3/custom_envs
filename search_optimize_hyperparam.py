'''Run an multi agent experiment.'''
import argparse
import logging
import os
from functools import partial
from itertools import count
from pathlib import Path

import gym
import tensorflow as tf
from tqdm import tqdm

import custom_envs.utils.utils_file as utils_file
import optuna
from custom_envs.utils.utils_logging import Monitor
from custom_envs.vectorize.optvecenv import OptVecEnv
from stable_baselines import A2C, PPO2
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.policies import MlpPolicy

LOGGER = logging.getLogger(__name__)


def get_total_reward(environment):
    '''
    Get the total reward so far from training.

    :param environment: () An environment wrapped in a Monitor Wrapper.
    '''
    return sum(sum(r) for r in environment.env_method('get_episode_rewards'))


def run_agent(envs, parameters, trial):
    '''Train an agent.'''
    path = Path(parameters['path'])
    dummy_env = OptVecEnv(envs)
    set_global_seeds(parameters.setdefault('seed'))
    if parameters['alg'] == 'PPO':
        model = PPO2(
            MlpPolicy, dummy_env, gamma=parameters['gamma'],
            learning_rate=parameters['learning_rate'], verbose=0
        )
    elif parameters['alg'] == 'A2C':
        model = A2C(
            MlpPolicy, dummy_env, gamma=parameters['gamma'],
            learning_rate=parameters['learning_rate'], verbose=0
        )
    try:
        timesteps = parameters['total_timesteps'] * dummy_env.agent_no_list[0]
        with tqdm(count(), leave=True) as progress:
            progress = iter(progress)

            def callback(local_vars, global_vars):
                if next(progress) % 100:
                    callback_env = local_vars['self'].env
                    get_total_reward(callback_env)
                    trial.report(
                        get_total_reward(callback_env), local_vars['update'])
                    if trial.should_prune():
                        raise optuna.structs.TrialPruned()

            model.learn(total_timesteps=timesteps, callback=callback)
        return get_total_reward(dummy_env)
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
            # 5 points to search
            'max_history': int(trial.suggest_discrete_uniform(
                'max_history', 5, 25, 5
            )),
            'max_batches': 100
        }
    }
    return parameters


def run_experiment(parameters, trial):
    '''Set up and run an experiment.'''
    parameters = parameters.copy()
    parameters.update(get_parameters(trial))
    trial_path = Path(parameters['path'], str(trial.number))
    trial_path.mkdir()
    parameters['path'] = str(trial_path)
    parameters['commit'] = utils_file.get_commit_hash(Path(__file__).parent)
    log_path = str(trial_path / 'monitor_{:d}')
    kwargs = parameters.setdefault('kwargs', {})
    utils_file.save_json(parameters, trial_path / 'parameters.json')
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
    parser.add_argument('--total_timesteps', default=int(5e6), type=int,
                        help="Number of timesteps per training session")
    parser.add_argument('--problem', help='The problem to optimize for.',
                        default='nn')
    parser.add_argument('--trials', help="The number of trials to run.",
                        default=10, type=int)
    args = parser.parse_args()
    parameters = vars(args).copy()
    del parameters['trials']
    path = Path(parameters['path'])
    if not path.exists():
        path.mkdir()
        parameters['kwargs'] = {'problem': parameters['problem']}
        utils_file.save_json(parameters, path / 'parameters.json')
    else:
        if (path / 'study.db').exists():
            print('Directory exists. Using existing study and parameters.')
            parameters = utils_file.load_json(path / 'parameters.json')
        else:
            raise FileExistsError(('Directory already exists and is not a '
                                   'study.'))
    objective = partial(run_experiment, parameters)
    storage = f'sqlite:///{path / "study.db"}'  # + str(path / 'study.db')
    study = optuna.create_study(
        study_name=str(path.name),
        storage=storage, load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=args.trials)


if __name__ == '__main__':
    main()
