
import os
import subprocess
from pathlib import Path
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

import custom_envs

def plot(axis_obj, sequence, **kwargs):
    axis_obj.plot(range(len(sequence)), sequence, **kwargs)

def fill_between(axis_obj, mean, std, **kwargs):
    axis_obj.fill_between(range(len(mean)), mean + std, mean - std, **kwargs)

def add_legend(axis):
    chartBox = axis.get_position()
    axis.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.8, chartBox.height])
    axis.legend(loc='upper center', bbox_to_anchor=(1.2, 0.8), shadow=True, ncol=1)


def task(path, seed):
    alg, learning_rate, gamma, _ = path.name.split('-')
    learning_rate = float(learning_rate)
    gamma = float(gamma)
    #env_name = 'OptimizeCorrect-v0'
    env_name = 'OptLRs-v0'
    save_path = path / 'model.pkl'
    env = gym.make(env_name, data_set='mnist', batch_size=32, version=3)
    #env = gym.make(env_name)
    if alg == 'PPO':
        with open(save_path, 'rb') as pkl:
            model = PPO2.load(pkl)
        #model = PPO2.load(save_path)
        alg_num = 0
    elif alg == 'A2C':
        with open(save_path, 'rb') as pkl:
            model = A2C.load(pkl)
        alg_num = 1
    elif alg == 'DDPG':
        model = DDPG.load(save_path)
        alg_num = 2
    env.seed(seed)
    obs = env.reset()
    terminal = False
    infos = []
    while not terminal:
    #for _ in range(75000):
    #for _ in range(1800):
        action, *_ = model.predict(obs)
        obs, reward, terminal, info = env.step(action)
        info['step'] = len(infos)
        info['reward'] = reward
        info['seed'] = seed
        infos.append(info)
    return infos

def task_lr():
    import tensorflow as tf
    import tensorflow.keras as keras

    from custom_envs import load_data
    from custom_envs.utils import to_onehot
    features, labels = load_data('iris')

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(3, input_shape=features.shape[1:],
              activation='softmax'))
    model.compile(tf.train.GradientDescentOptimizer(6e-2),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    hist = model.fit(features, labels, epochs=39, verbose=1)
    print(hist.history['acc'], hist.history['loss'])
    return hist.history['acc'], hist.history['loss'] 

def run_multi():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="The directory to save files to.")
    args = parser.parse_args()
    path = Path(args.path)
    infos = list(chain.from_iterable([task(path, i) for i in trange(1)]))
    dataframe = pd.DataFrame(infos)
    mean = dataframe.groupby('step').mean()
    std = dataframe.groupby('step').std()
    #print(dataframe)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Performance on SKIN data set')
    ax.set_ylabel('loss')
    ax.set_xlabel('step')
    plot(ax, mean['objective'])
    fill_between(ax, mean['objective'], std['objective'], alpha=0.1)
    add_legend(ax)



    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Performance on SKIN data set')
    ax.set_ylabel('accuracy')
    ax.set_xlabel('step')
    plot(ax, mean['accuracy'])
    print(mean['accuracy'])
    fill_between(ax, mean['accuracy'], std['accuracy'], alpha=0.1)
    add_legend(ax)

    plt.show()

    dataframe.to_csv('skin_results.csv')


if __name__ == '__main__':
    run_multi()
    #task_lr()
