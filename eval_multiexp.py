'''Module for evaluating learned agents against different environments.'''
import argparse
from threading import Thread
from pathlib import Path
from functools import partial
from itertools import chain

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange
from stable_baselines import PPO2, A2C, DDPG

import custom_envs.utils.utils_plot as utils_plot
from custom_envs.multiagent import MultiEnvServer
from custom_envs.utils.utils_venv import ThreadVecEnv
from custom_envs.envs.multioptlrs import MultiOptLRs
from custom_envs.utils.utils_common import enzip
from custom_envs.data import load_data


def run_handle(env):
    '''Run handle requests until complete.'''
    data = 0
    while data is not None:
        data = env.handle_requests()


def task(path, seed, batch_size=None):
    '''Run the agent on a data set.'''
    alg, *_ = path.name.split('-')
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
    states = dummy_env.reset()
    info_list = []
    epoch_no = 0
    current_sample_no = 0
    while epoch_no < 40:
        action, *_ = model.predict(states)
        states, rewards, _, infos = dummy_env.step(action)
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


def task_lr(seed, batch_size=None):
    '''Train a logistic classification model.'''
    sequence = load_data('mnist')
    features = sequence.features
    labels = sequence.labels
    batch_size = len(features) if batch_size is None else batch_size
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(labels.shape[-1],
                                    input_shape=features.shape[1:],
                                    activation='softmax'))
    model.compile(tf.train.GradientDescentOptimizer(0.1),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    hist = model.fit(features, labels, epochs=40, verbose=0,
                     batch_size=batch_size).history
    return [{'epoch': epoch, 'loss': lss, 'accuracy': acc, 'seed': seed}
            for epoch, lss, acc in enzip(hist['loss'], hist['acc'])]


def run_multi(trials=10, batch_size=None):
    '''Run both agent evaluationg and logistic classification training.'''
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="The directory to save files to.")
    args = parser.parse_args()
    path = Path(args.path)
    infos = list(chain.from_iterable([task(path, i, batch_size=batch_size)
                                      for i in trange(trials)]))
    dataframe_rl = pd.DataFrame(infos)
    infos = list(chain.from_iterable([task_lr(i, batch_size=batch_size)
                                      for i in trange(trials)]))
    dataframe_lc = pd.DataFrame(infos)
    mean_rl = dataframe_rl.groupby('epoch').mean()
    std_rl = dataframe_rl.groupby('epoch').std()
    mean_lc = dataframe_lc.groupby('epoch').mean()
    std_lc = dataframe_lc.groupby('epoch').std()
    fig = plt.figure()
    axis = fig.add_subplot(111)
    pyplot_attr = {
        'title': 'Performance on MNIST data set',
        'ylabel': 'Loss',
        'xlabel': 'Epoch',
    }
    utils_plot.set_attributes(axis, pyplot_attr)
    utils_plot.plot_sequence(axis, mean_rl['objective'], label='RL')
    utils_plot.fill_between(axis, mean_rl['objective'], std_rl['objective'],
                            alpha=0.1, label='RL')
    utils_plot.plot_sequence(axis, mean_lc['loss'],
                             label='Logistic Classification')
    utils_plot.fill_between(axis, mean_lc['loss'], std_lc['loss'], alpha=0.1,
                            label='Logistic Classification')
    utils_plot.add_legend(axis)

    fig = plt.figure()
    axis = fig.add_subplot(111)
    pyplot_attr['ylabel'] = 'Accuracy'
    utils_plot.set_attributes(axis, pyplot_attr)
    utils_plot.plot_sequence(axis, mean_rl['accuracy'], label='RL')
    utils_plot.fill_between(axis, mean_rl['accuracy'], std_rl['accuracy'],
                            alpha=0.1, label='RL')
    utils_plot.plot_sequence(axis, mean_lc['accuracy'],
                             label='Logistic Classification')
    utils_plot.fill_between(axis, mean_lc['accuracy'], std_lc['accuracy'],
                            alpha=0.1, label='Logistic Classification')
    utils_plot.add_legend(axis)
    plt.show()


if __name__ == '__main__':
    run_multi(1, batch_size=None)
