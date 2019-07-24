'''Module for evaluating learned agents against different environments.'''
import argparse
import os
from math import ceil
from pathlib import Path
from functools import partial
from itertools import chain
from collections import defaultdict

import gym
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from stable_baselines import PPO2, A2C, DDPG

import custom_envs.utils.utils_plot as utils_plot
from custom_envs.multiagent import MultiEnvServer
from custom_envs.utils.utils_venv import ThreadVecEnv
from custom_envs.envs.multioptimize import MultiOptimize
from custom_envs.envs.multioptlrs import MultiOptLRs
from custom_envs.utils.utils_common import enzip
from custom_envs.data import load_data
from custom_envs.vectorize import OptVecEnv
import custom_envs.utils.utils_file as utils_file


def flatten_arrays(arrays):
    return list(chain.from_iterable(a.ravel().tolist() for a in arrays))


class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        self.history.append({
            'epoch': epoch,
            'weights_mean': np.mean(flatten_arrays(self.model.get_weights())),
            **logs
        })


def run_handle(env):
    '''Run handle requests until complete.'''
    data = 0
    while data is not None:
        data = env.handle_requests()


def task(path, seed, batch_size=None, total_epochs=40, data_set='mnist'):
    '''
    Run the agent on a data set.
    '''
    parameters = utils_file.load_json(path / 'hyperparams.json')
    alg, *_ = path.name.split('-')
    save_path = path / 'model.pkl'
    sequence = load_data(data_set)
    num_of_samples = len(sequence.features)
    steps_per_epoch = ceil(num_of_samples / batch_size) if batch_size else 1
    kwargs = parameters.get('kwargs', {})
    kwargs['data_set'] = data_set
    kwargs['batch_size'] = batch_size
    kwargs['max_batches'] = steps_per_epoch*total_epochs
    env = partial(gym.make, parameters['env_name'], **kwargs)
    vec_env = OptVecEnv([env])
    if alg == 'PPO':
        with open(save_path, 'rb') as pkl:
            model = PPO2.load(pkl)  # , env=vec_env)
    elif alg == 'A2C':
        with open(save_path, 'rb') as pkl:
            model = A2C.load(pkl, env=vec_env)
    elif alg == 'DDPG':
        model = DDPG.load(save_path, env=vec_env)
    states = vec_env.reset()
    info_list = []
    cumulative_reward = 0
    for epoch_no in trange(total_epochs, leave=False):
        for step in trange(steps_per_epoch, leave=False):
            actions = model.predict(states, deterministic=False)[0]
            states, rewards, _, infos = vec_env.step(actions)
            cumulative_reward = cumulative_reward + rewards[0]
            info = infos[0]
            info['step'] = epoch_no*steps_per_epoch + step
            info['cumulative_reward'] = cumulative_reward
            info['seed'] = seed
            info['epoch'] = epoch_no
            info_list.append(info)
    return info_list


def task_lr(seed, batch_size=None, total_epochs=40, data_set='mnist'):
    '''Train a logistic classification model.'''
    sequence = load_data(data_set)
    features = sequence.features
    labels = sequence.labels
    batch_size = len(features) if batch_size is None else batch_size
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(
        48, input_shape=features.shape[1:],
        kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed),
        bias_initializer=tf.keras.initializers.glorot_normal(seed=seed),
        activation='relu', use_bias=True
    ))
    model.add(tf.keras.layers.Dense(
        48,
        kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed),
        bias_initializer=tf.keras.initializers.glorot_normal(seed=seed),
        activation='relu', use_bias=True
    ))
    model.add(tf.keras.layers.Dense(
        labels.shape[-1],
        kernel_initializer=tf.keras.initializers.glorot_normal(seed=seed),
        bias_initializer=tf.keras.initializers.glorot_normal(seed=seed),
        activation='softmax')
    )
    # model.compile(tf.train.GradientDescentOptimizer(1e-1),
    #              loss='categorical_crossentropy',
    #              metrics=['accuracy'])
    model.compile(tf.train.AdamOptimizer(1e-2),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    callback = CustomCallback()
    model.fit(features, labels, epochs=total_epochs, verbose=0, shuffle=True,
              batch_size=batch_size, callbacks=[callback])
    return callback.history


def plot_results(axes, dataframe, groupby, label=None):
    '''Plot results on multiple axes given a dataframe.'''
    grouped = dataframe.groupby(groupby)
    mean_df = grouped.mean()
    std_df = grouped.std()
    columns = set(mean_df.columns) & set(axes.keys()) - {groupby}
    for name in columns:
        utils_plot.plot_sequence(axes[name], mean_df[name], label=label)
        utils_plot.fill_between(axes[name], mean_df[name], std_df[name],
                                alpha=0.1, label=label)


def run_multi(path, trials=10, batch_size=None, total_epochs=40,
              data_set='mnist'):
    '''Run both agent evaluationg and logistic classification training.'''
    path = Path(path)
    infos = list(chain.from_iterable([task(path, i, batch_size=batch_size,
                                           total_epochs=total_epochs,
                                           data_set=data_set)
                                      for i in trange(trials)]))
    dataframe_rl = pd.DataFrame(infos)
    infos = list(chain.from_iterable([task_lr(i, batch_size=batch_size,
                                              total_epochs=total_epochs,
                                              data_set=data_set)
                                      for i in trange(trials)]))
    dataframe_lc = pd.DataFrame.from_dict(infos)
    columns = ['accuracy' if col == 'acc' else col
               for col in dataframe_lc.columns]
    dataframe_lc.columns = columns
    axes = defaultdict(lambda: plt.figure().add_subplot(111))
    pyplot_attr = {
        'title': 'Performance on {} data set'.format(data_set.upper()),
        'xlabel': 'Epoch',
    }
    columns = set(dataframe_rl.select_dtypes('number').columns) - {'epoch'}
    for column in columns:
        pyplot_attr['ylabel'] = column.capitalize()
        utils_plot.set_attributes(axes[column], pyplot_attr)

    plot_results(axes, dataframe_rl, 'epoch', 'RL')
    plot_results(axes, dataframe_lc, 'epoch', 'Logistic Classification')
    for axis in axes.values():
        utils_plot.add_legend(axis)
    plt.show()


def run_lr(path, trials=10, batch_size=None, total_epochs=40,
           data_set='mnist'):
    '''Run both agent evaluationg and logistic classification training.'''
    path = Path(path)
    infos = list(chain.from_iterable([task_lr(i, batch_size=batch_size,
                                              total_epochs=total_epochs,
                                              data_set=data_set)
                                      for i in trange(trials)]))
    dataframe_lc = pd.DataFrame.from_dict(infos)
    columns = ['accuracy' if col == 'acc' else col
               for col in dataframe_lc.columns]
    dataframe_lc.columns = columns
    axes = defaultdict(lambda: plt.figure().add_subplot(111))
    pyplot_attr = {
        'title': 'Performance on {} data set'.format(data_set.upper()),
        'xlabel': 'Epoch',
    }
    # columns = set(dataframe_rl.select_dtypes('number').columns) - {'epoch'}
    for column in columns:
        pyplot_attr['ylabel'] = column.capitalize()
        utils_plot.set_attributes(axes[column], pyplot_attr)
    dataframe_lc.columns = columns

    plot_results(axes, dataframe_lc, 'epoch', 'Logistic Classification')
    for axis in axes.values():
        utils_plot.add_legend(axis)
    plt.show()


def main():
    '''Evaluate a trained model against logistic regression.'''
    parser = argparse.ArgumentParser()
    parser.add_argument("model_weights", help="The path to the model weights.",
                        type=Path)
    parser.add_argument("--trials", help="The number of trials.",
                        type=int, default=1)
    parser.add_argument("--batch_size", help="The batch size.",
                        type=int, default=32)
    parser.add_argument("--total_epochs", help="The number of epochs.",
                        type=int, default=40)
    parser.add_argument("--data_set", help="The data set to trial against.",
                        type=str, default='iris')
    args = parser.parse_args()
    tf.logging.set_verbosity(tf.logging.ERROR)
    run_lr(args.model_weights, args.trials, args.batch_size,
           args.total_epochs, args.data_set)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
