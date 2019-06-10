import os
from concurrent.futures import ProcessPoolExecutor
from math import log10
from random import uniform
from collections import namedtuple
from itertools import zip_longest, product, chain

import numpy as np
import pandas as pd
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.contrib.opt as tf_opt
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt

from custom_envs import load_data
from custom_envs.utils import to_onehot

from tqdm import tqdm

def max_group(dataframe, by, column, method='mean'):
    groups = dataframe.groupby(by)
    if method == 'mean':
        return groups.mean()[column].idxmax()

def min_group(dataframe, by, column, method='mean'):
    groups = dataframe.groupby(by)
    if method == 'mean':
        return groups.mean()[column].idxmin()

def plot(axis_obj, sequence, **kwargs):
    axis_obj.plot(range(len(sequence)), sequence, **kwargs)

def fill_between(axis_obj, mean, std, **kwargs):
    axis_obj.fill_between(range(len(mean)), mean + std, mean - std, **kwargs)

def add_legend(axis):
    chartBox = axis.get_position()
    axis.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.8, chartBox.height])
    axis.legend(loc='upper center', bbox_to_anchor=(1.2, 0.8), shadow=True, ncol=1)

def run(args):
    seed, lr = args
    tf.set_random_seed(seed)
    samples, labels = load_data()
    image_size = samples.shape[-1:]

    samples = samples.astype(np.float32)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(3, input_shape=image_size,
                                    activation='softmax', use_bias=False))
    model.compile(tf.train.GradientDescentOptimizer(lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    rows = []
    for i in range(40):
        hist = model.fit(samples, labels, epochs=1, verbose=0).history
        rows.append({'step': i, 'loss': hist['loss'][0], 'seed': seed,
                     'acc': hist['acc'][0], 'learning_rate': lr})
    return rows

def main():
    num_of_seed = 10
    learning_rates = 10**np.linspace(-1, -2, num=10)
    with ProcessPoolExecutor() as executor:
        tasks = executor.map(run, product(range(num_of_seed), learning_rates))
        tasks = tqdm(tasks, total=num_of_seed*len(learning_rates))
        tasks = chain.from_iterable(tasks)
        dataframe = pd.DataFrame(list(tasks))

    figure_acc = plt.figure()
    axis_acc = figure_acc.add_subplot(111)
    figure_loss = plt.figure()
    axis_loss = figure_loss.add_subplot(111)
    group_df = dataframe.groupby('learning_rate')

    for name, group in group_df:
        aggregate = group.groupby('step')
        mean = aggregate.mean()
        std = aggregate.std()
        plot(axis_acc, mean['acc'], label='LR={:f}'.format(name))
        fill_between(axis_acc, mean['acc'], std['acc'], alpha=0.1)
        plot(axis_loss, mean['loss'], label='LR={:f}'.format(name))
        fill_between(axis_loss, mean['loss'], std['loss'], alpha=0.1)
    add_legend(axis_acc)
    add_legend(axis_loss)
    figure_acc.savefig('linreg_different_LR_ACC.png')
    figure_loss.savefig('linreg_different_LR_LOSS.png')
    plt.show()

    dataframe.to_csv('logcls_results.csv')

def main2():
    print(pd.DataFrame(run((0, 7e-2))))

if __name__ == '__main__':
    main()
