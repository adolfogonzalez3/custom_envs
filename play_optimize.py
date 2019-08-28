'''Run an multi agent experiment.'''
import os
import logging
import argparse
from pathlib import Path
from functools import partial
from itertools import count
from collections.abc import Callable

import gym
import optuna
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm, trange
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines import PPO2, A2C
from stable_baselines.common.misc_util import set_global_seeds

import custom_envs.utils.utils_file as utils_file
import custom_envs.utils.utils_common as utils_common
from custom_envs.utils.utils_logging import Monitor
from custom_envs.vectorize.optvecenv import OptVecEnv
from custom_envs.utils.utils_functions import compute_rosenbrock

LOGGER = logging.getLogger(__name__)

class Animate(Callable):
    def __init__(self, axis):
        self.axis = axis
        self.artists = []
        self.positions = []

    def __call__(self, frame, *fargs):
        if self.artists:
            self.artists.pop().remove()
        self.positions.append(frame)
        pos_x, pos_y = list(zip(*self.positions))
        self.artists.append(self.axis.plot(pos_x, pos_y, 'r')[0])
        return self.artists[-1]

def live_plot(optimize, name):
    discrete_points = 100  # number of discretization points along both axes
    x_range = (-2., 2.)
    y_range = (-1.5, 4.)

    x_points, y_points = np.meshgrid(
        np.linspace(*x_range, discrete_points),
        np.linspace(*y_range, discrete_points)
    )
    z_points = compute_rosenbrock(x_points, y_points)

    fig = plt.figure()
    plt.rc('font', family='serif')
    axis = fig.add_subplot(1, 1, 1)
    axis.contour(
        x_points, y_points, z_points,
        np.logspace(-0.5, 3.5, 10, base=10), cmap='gray'
    )
    axis.set_title(
        r'$\mathrm{Rosenbrock Function}: f(x,y)=(1-x)^2+100(y-x^2)^2$'
    )
    axis.set_xlim(*x_range)
    axis.set_ylim(y_range)
    axis.set_xlabel('x')
    axis.set_ylabel('y')

    
    ani = animation.FuncAnimation(
        fig, Animate(axis), frames=optimize, interval=100, repeat=False
    )
    ani.save('{}.gif'.format(name))
    plt.show()

def run_agent(envs, parameters):
    '''Train an agent.'''
    path = Path(parameters['path'])
    dummy_env = OptVecEnv(envs)
    set_global_seeds(parameters.setdefault('seed'))
    save_path = str(path / 'model.pkl')
    alg = parameters['alg']
    if alg == 'PPO':
        with open(save_path, 'rb') as pkl:
            model = PPO2.load(pkl, env=dummy_env)
    elif alg == 'A2C':
        with open(save_path, 'rb') as pkl:
            model = A2C.load(pkl, env=dummy_env)
    try:
        done = False
        observations = dummy_env.reset()
        while not done:
            action = model.predict(observations)
            print(action[0].ravel().tolist())
            observations, rewards, dones, infos = dummy_env.step(action[0])
            done = any(dones)
            info = infos[0]
            yield info['weights']
    finally:
        dummy_env.close()


def run_experiment(parameters):
    '''Set up and run an experiment.'''
    wrapped_envs = [
        partial(
            gym.make, parameters['env_name'], **parameters['kwargs']
        ) for i in range(1)
    ]
    yield from run_agent(wrapped_envs, parameters)
    #run_agent(wrapped_envs, parameters)


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
    storage = 'sqlite:///' + str(path / 'study.db')
    study = optuna.create_study(
        study_name=str(path.name),
        storage=storage, load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )
    print(study.best_value)
    print(study.best_trial.number)
    print(study.best_params)
    #params = study.best_params
    param_file = path / str(study.best_trial.number) / 'parameters.json'
    params = utils_file.load_json(param_file)
    kwargs = params['kwargs']
    parameters.update({
        'path': path / str(study.best_trial.number),
        'gamma': params['gamma'],
        'learning_rate': params['learning_rate'],
        'kwargs': {
            #'batch_size': int(params['batch_size']),
            'max_batches': int(kwargs['max_batches']),
            'max_history': int(kwargs['max_history']),
            'max_batches': 1e3
        }
    })
    live_plot(run_experiment(parameters), 'test')
    #run_experiment(parameters)

if __name__ == '__main__':
    main()
