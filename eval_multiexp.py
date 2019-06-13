'''Module for evaluating learned agents against different environments.'''
import math
import argparse
from threading import Thread
from pathlib import Path
from functools import partial
from itertools import chain
from collections import defaultdict

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


def run_handle(env):
    '''Run handle requests until complete.'''
    data = 0
    while data is not None:
        data = env.handle_requests()


def task(path, seed, batch_size=None, total_epochs=40, data_set='mnist'):
    '''Run the agent on a data set.'''
    alg, *_ = path.name.split('-')
    save_path = path / 'model.pkl'
    #env = MultiOptLRs(data_set=data_set, batch_size=batch_size)
    env = MultiOptimize(data_set=data_set, batch_size=batch_size)
    sequence = load_data(data_set)
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
    if batch_size is not None:
        steps_per_epoch = math.ceil(num_of_samples / batch_size)
    else:
        steps_per_epoch = 1
    for epoch_no in trange(total_epochs):
        for step in trange(steps_per_epoch):
            action, *_ = model.predict(states)
            states, rewards, _, infos = dummy_env.step(action)
            info = infos[0]
            info['step'] = epoch_no*steps_per_epoch + step
            info['reward'] = sum(rewards)
            info['seed'] = seed
            info['epoch'] = epoch_no
            info_list.append(info)
    dummy_env.close()
    taskrun.join()
    return info_list


def task_lr(seed, batch_size=None, total_epochs=40, data_set='mnist'):
    '''Train a logistic classification model.'''
    sequence = load_data(data_set)
    features = sequence.features
    labels = sequence.labels
    batch_size = len(features) if batch_size is None else batch_size
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(labels.shape[-1],
                                    input_shape=features.shape[1:],
                                    activation='softmax'))
    model.compile(tf.train.GradientDescentOptimizer(1e-1),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    hist = model.fit(features, labels, epochs=total_epochs, verbose=0,
                     batch_size=batch_size).history
    return [{'epoch': epoch, 'loss': lss, 'accuracy': acc, 'seed': seed}
            for epoch, lss, acc in enzip(hist['loss'], hist['acc'])]


def plot_results(axes, dataframe, groupby, targets, label=None):
    grouped = dataframe.groupby(groupby)
    mean_df = grouped.mean()
    std_df = grouped.std()
    for axis, target in zip(axes, targets):
        utils_plot.plot_sequence(axis, mean_df[target], label=label)
        utils_plot.fill_between(axis, mean_df[target], std_df[target],
                                alpha=0.1, label=label)


def run_multi(path, trials=10, batch_size=None, total_epochs=40,
              data_set='mnist'):
    '''Run both agent evaluationg and logistic classification training.'''
    path = Path(path)
    if True:
        infos = list(chain.from_iterable([task(path, i, batch_size=batch_size,
                                               total_epochs=total_epochs,
                                               data_set=data_set)
                                          for i in trange(trials)]))
        dataframe_rl = pd.DataFrame(infos)
    infos = list(chain.from_iterable([task_lr(i, batch_size=batch_size,
                                              total_epochs=total_epochs,
                                              data_set=data_set)
                                      for i in trange(trials)]))
    dataframe_lc = pd.DataFrame(infos)
    axes = defaultdict(lambda: plt.figure().add_subplot(111))
    pyplot_attr = {
        'title': 'Performance on {} data set'.format(data_set.upper()),
        'ylabel': 'Loss',
        'xlabel': 'Epoch',
    }
    utils_plot.set_attributes(axes['loss'], pyplot_attr)
    pyplot_attr['ylabel'] = 'Accuracy'
    utils_plot.set_attributes(axes['accuracy'], pyplot_attr)
    plot_results(axes.values(), dataframe_rl, 'epoch',
                 ['objective', 'accuracy'], 'RL')
    plot_results(axes.values(), dataframe_lc, 'epoch', ['loss', 'accuracy'],
                 'Logistic Classification')
    utils_plot.add_legend(axes['loss'])
    utils_plot.add_legend(axes['accuracy'])
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
    run_multi(args.model_weights, args.trials, args.batch_size,
              args.total_epochs, args.data_set)


if __name__ == '__main__':
    main()
