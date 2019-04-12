
import os
import subprocess
from collections import Counter, defaultdict
from itertools import product, cycle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import gym
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C, DDPG
from stable_baselines.common.misc_util import set_global_seeds


def task(args):
    env_name, seed, path = args
    log_dir = os.path.join(path, env_name, 'ppo-{:d}'.format(seed))
    save_path = os.path.join(path, env_name, 'ppo-{:d}'.format(seed), 'model')
    env = gym.make(env_name)
    env.seed(seed)
    set_global_seeds(seed)

    model = PPO2.load(save_path)
    obs = env.reset()
    terminal = False
    accuracies = []
    objectives = []
    while not terminal:
        action, *_ = model.predict(obs)
        obs, _, terminal, info = env.step(action)
        accuracies.append(info['accuracy'])
        objectives.append(info['objective'])
    return accuracies, objectives

def task_lr(*args):
    import tensorflow as tf
    import tensorflow.keras as keras

    from custom_envs.load_data import load_data
    from custom_envs.utils import to_onehot
    tf.set_random_seed(args[0])
    data = load_data()
    features = data[:, :-1]
    labels = data[:, -1:]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(3, input_shape=features.shape[1:], activation='softmax'))
    model.compile(tf.train.GradientDescentOptimizer(1e-2),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    hist = model.fit(features, labels, epochs=39, verbose=0)
    return hist.history['acc'], hist.history['loss']

def get_experiments(path):
    experiments = os.listdir(path)
    expr_names = [e.rsplit('-', 1)[0] for e in experiments]
    return Counter(expr_names)
    

def run():
    import argparse
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("path", help="The directory to save files to.")
    ARGS = PARSER.parse_args()
    PATH = [ARGS.path]
    ENV_NAMES = os.listdir(ARGS.path)
    with ProcessPoolExecutor() as executor:
        results = []
        for env_name in ENV_NAMES:
            results.append(executor.map(task, product([env_name], range(10), PATH)))
        lr_result = executor.map(task_lr, range(10))
        results = [list(rs) for rs in results]
        lr_result = list(lr_result)
    for env_name, result in zip(ENV_NAMES, results):
        accuracies, objectives = list(zip(*result))
        accuracies = np.array(accuracies)
        objectives = np.array(objectives)
        mean_obj = np.mean(objectives, axis=0)
        std_obj = np.std(objectives, axis=0)
        plt.plot(range(len(mean_obj)), mean_obj, label=env_name)
        #plt.ylim(0,1)
        plt.fill_between(range(len(mean_obj)), mean_obj - std_obj,
                         mean_obj+std_obj, alpha=0.1)
        #plt.title(env_name)
    accuracies, objectives = list(zip(*lr_result))
    accuracies = np.array(accuracies)
    objectives = np.array(objectives)
    mean_obj = np.mean(objectives, axis=0)
    std_obj = np.std(objectives, axis=0)
    plt.plot(range(len(mean_obj)), mean_obj, label='LR minibatch')
    #plt.ylim(0, 1)
    plt.fill_between(range(len(mean_obj)), mean_obj - std_obj,
                     mean_obj+std_obj, alpha=0.1)
    plt.legend()
    plt.title('Training for 39 epochs on the IRIS data set')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.show()

def task2(args):
    (folder_name_partial, seeds), path = args
    alg, _, learning_rate, _, gamma = folder_name_partial.split('_')
    learning_rate = float(learning_rate)
    gamma = float(gamma)
    dataframes = []
    for seed in range(seeds):
        folder_name = '-'.join([folder_name_partial, str(seed)])
        env_name = 'Optimize-v0'
        alg = folder_name_partial.split('_', 1)[0]
        log_dir = os.path.join(path, env_name, folder_name)
        save_path = os.path.join(path, env_name, folder_name, 'model')
        env = gym.make(env_name)
        env.seed(seed)
        set_global_seeds(seed)
        alg_num = -1
        if alg == 'PPO':
            model = PPO2.load(save_path)
            alg_num = 0
        elif alg == 'A2C':
            model = A2C.load(save_path)
            alg_num = 1
        elif alg == 'DDPG':
            model = DDPG.load(save_path)
            alg_num = 2
        else:
            print('ERROR: ', alg)
        obs = env.reset()
        terminal = False

        data = defaultdict(list)
        while not terminal:
            action, *_ = model.predict(obs)
            obs, _, terminal, info = env.step(action)
            data['accuracy'].append(info['accuracy'])
            data['objective'].append(info['objective'])
        data['seed'] = [seed]*len(data['accuracy'])
        data['learning_rate'] = [learning_rate]*len(data['accuracy'])
        data['gamma'] = [gamma]*len(data['accuracy'])
        data['alg'] = [alg_num]*len(data['accuracy'])
        dataframes.append(pd.DataFrame(data))
    return pd.concat(dataframes)

def run_multi():
    import argparse
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("path", help="The directory to save files to.")
    ARGS = PARSER.parse_args()
    PATH = [ARGS.path]
    ENV_NAMES = os.listdir(ARGS.path)
    env_expr_dicts = [get_experiments(os.path.join(ARGS.path, env_name))
                      for env_name in ENV_NAMES]
    with ProcessPoolExecutor() as executor:
        env_tasks = [
            executor.map(task2, zip(expr_dict.items(), cycle(PATH)))
            for env_name, expr_dict in zip(ENV_NAMES, env_expr_dicts)
        ]
        lr_result = executor.map(task_lr, range(3))
        results = [list(rs) for rs in env_tasks]
        lr_result = list(lr_result)
    for env_name, result in zip(ENV_NAMES, results):
        result_df = pd.concat(result)
        result_df.reset_index(inplace=True)
        result_df.to_csv('results_{}.csv'.format(env_name))


if __name__ == '__main__':
    run_multi()