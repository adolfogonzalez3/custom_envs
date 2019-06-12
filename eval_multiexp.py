
import os
import subprocess
from threading import Thread
from pathlib import Path
from functools import partial
from collections import Counter, defaultdict
from itertools import product, cycle, chain
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import trange

import gym
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C, DDPG
from stable_baselines.common.misc_util import set_global_seeds

from custom_envs.multiagent import EnvironmentInSync, MultiEnvServer
from custom_envs.utils.utils_venv import SubprocVecEnv, ThreadVecEnv
from custom_envs.envs.multioptlrs import MultiOptLRs

import custom_envs
from custom_envs.utils.utils_common import enzip
from custom_envs.data import load_data


def plot(axis_obj, sequence, **kwargs):
    axis_obj.plot(range(len(sequence)), sequence, **kwargs)


def fill_between(axis_obj, mean, std, **kwargs):
    axis_obj.fill_between(range(len(mean)), mean + std, mean - std, **kwargs)


def add_legend(axis):
    chartBox = axis.get_position()
    axis.set_position(
        [chartBox.x0, chartBox.y0, chartBox.width*0.8, chartBox.height])
    axis.legend(loc='upper center', bbox_to_anchor=(
        1.2, 0.8), shadow=True, ncol=1)


def run_handle(env):
    data = 0
    while data is not None:
        data = env.handle_requests()


def task(path, seed, batch_size=None):
    alg, learning_rate, gamma, _ = path.name.split('-')
    learning_rate = float(learning_rate)
    gamma = float(gamma)
    env_name = 'MultiOptLRs-v0'
    save_path = path / 'model.pkl'
    env = MultiOptLRs(data_set='mnist', batch_size=batch_size)
    sequence = load_data('mnist')
    num_of_samples = len(sequence.features)
    main_environment = MultiEnvServer(env)

    envs = [partial(lambda x: x, subenv) for subenv in
            main_environment.sub_environments.values()]
    dummy_env = ThreadVecEnv(envs)
    if alg == 'PPO':
        with open(save_path, 'rb') as pkl:
            model = PPO2.load(pkl)
    elif alg == 'A2C':
        with open(save_path, 'rb') as pkl:
            model = A2C.load(pkl)
    elif alg == 'DDPG':
        model = DDPG.load(save_path)
    taskrun = Thread(target=run_handle, args=[main_environment])
    taskrun.start()
    # dummy_env.seed(seed)
    obs = dummy_env.reset()
    terminals = [False]
    info_list = []
    epoch_no = 0
    current_sample_no = 0
    while epoch_no < 40:
    #while not all(terminals):
    # for _ in range(75000):
    # for _ in range(1800):
        action, *_ = model.predict(obs)
        states, rewards, terminals, infos = dummy_env.step(action)
        info = infos[0]
        info['step'] = len(info_list)
        info['reward'] = sum(rewards)
        info['seed'] = seed
        info['epoch'] = epoch_no
        info_list.append(info)
        if batch_size is not None:
            current_sample_no += batch_size
            if current_sample_no >= num_of_samples:
                epoch_no += 1
                current_sample_no = 0
        else:
            epoch_no += 1
    dummy_env.close()
    taskrun.join()
    return info_list


def task_lr(seed):
    import tensorflow as tf
    import tensorflow.keras as keras

    from custom_envs import load_data
    sequence = load_data('mnist')
    features = sequence.features
    labels = sequence.labels

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(labels.shape[-1],
                                    input_shape=features.shape[1:],
                                    activation='softmax'))
    model.compile(tf.train.GradientDescentOptimizer(0.1),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    hist = model.fit(features, labels, epochs=40, verbose=0).history
    info = [{'epoch': epoch, 'loss': lss, 'accuracy': acc, 'seed': seed}
            for epoch, lss, acc in enzip(hist['loss'], hist['acc'])]


def run_multi(trials=10):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="The directory to save files to.")
    args = parser.parse_args()
    path = Path(args.path)
    infos = list(chain.from_iterable([task(path, i) for i in trange(trials)]))
    dataframe_rl = pd.DataFrame(infos)
    infos = list(chain.from_iterable([task_lr(i) for i in trange(trials)]))
    dataframe_lc = pd.DataFrame(infos)
    mean_rl = dataframe_rl.groupby('epoch').mean()
    std_rl = dataframe_rl.groupby('epoch').std()
    mean_lc = dataframe_lc.groupby('epoch').mean()
    std_lc = dataframe_lc.groupby('epoch').std()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Performance on MNIST data set')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    plot(ax, mean_rl['objective'], label='RL')
    fill_between(ax, mean_rl['objective'], std_rl['objective'], alpha=0.1,
                 label='RL')
    plot(ax, mean_lc['loss'], label='Logistic Classification')
    fill_between(ax, mean_lc['loss'], std_lc['loss'], alpha=0.1,
                 label='Logistic Classification')
    add_legend(ax)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Performance on MNIST data set')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('epoch')
    plot(ax, mean_rl['accuracy'], label='RL')
    fill_between(ax, mean_rl['accuracy'], std_rl['accuracy'], alpha=0.1,
                 label='RL')
    plot(ax, mean_lc['accuracy'], label='Logistic Classification')
    fill_between(ax, mean_lc['accuracy'], std_lc['accuracy'], alpha=0.1,
                 label='Logistic Classification')
    add_legend(ax)
    plt.show()


if __name__ == '__main__':
    run_multi(10)
