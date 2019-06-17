'''Module for evaluating learned agents against different environments.'''
import argparse
from math import ceil
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
    '''
    Run the agent on a data set.
    '''
    alg, *_ = path.name.split('-')
    save_path = path / 'model.pkl'
    sequence = load_data(data_set)
    num_of_samples = len(sequence.features)
    steps_per_epoch = ceil(num_of_samples / batch_size) if batch_size else 1
    # env = MultiOptLRs(data_set=data_set, batch_size=batch_size)
    env = MultiOptimize(data_set=data_set, batch_size=batch_size, version=1,
                        max_batches=steps_per_epoch*total_epochs)
    main_environment = MultiEnvServer(env)

    envs = [partial(lambda x: x, subenv) for subenv in
            main_environment.sub_environments.values()]
    dummy_env = ThreadVecEnv(envs)
    if alg == 'PPO':
        with open(save_path, 'rb') as pkl:
            model = PPO2.load(pkl, env=dummy_env)
    elif alg == 'A2C':
        with open(save_path, 'rb') as pkl:
            model = A2C.load(pkl, env=dummy_env)
    elif alg == 'DDPG':
        model = DDPG.load(save_path, env=dummy_env)
    taskrun = Thread(target=run_handle, args=[main_environment])
    taskrun.start()
    states = dummy_env.reset()
    info_list = []
    cumulative_reward = 0
    for epoch_no in trange(total_epochs, leave=False):
        for step in trange(steps_per_epoch, leave=False):
            action, *_ = model.predict(states)
            states, rewards, _, infos = dummy_env.step(action)
            cumulative_reward = cumulative_reward + rewards[0]
            info = infos[0]
            info['step'] = epoch_no*steps_per_epoch + step
            info['cumulative_reward'] = cumulative_reward
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
                                    activation='softmax',
                                    use_bias=True))
    model.compile(tf.train.GradientDescentOptimizer(1e-1),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    hist = model.fit(features, labels, epochs=total_epochs, verbose=0,
                     batch_size=batch_size, shuffle=True).history
    return [{'epoch': epoch, 'loss': lss, 'accuracy': acc, 'seed': seed}
            for epoch, lss, acc in enzip(hist['loss'], hist['acc'])]


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
    dataframe_lc = pd.DataFrame(infos)
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
