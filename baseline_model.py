import os
from concurrent.futures import ProcessPoolExecutor
from math import log10
from random import uniform
from collections import namedtuple
from itertools import zip_longest, product
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.opt as tf_opt
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt

from custom_envs import load_data

from tqdm import tqdm

def discretize(x):
    x = x.ravel()
    x_unique = np.unique(x)
    return np.argmax(x[:, None] == x_unique[None, :], axis=1)

def run(args):
    seed, lr = args
    tf.set_random_seed(seed)
    samples, labels = load_data()
    image_size = samples.shape[-1:]
    labels = discretize(labels)

    train_samples = samples.astype(np.float32)
    train_labels = labels
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(3, input_shape=image_size,
                                    activation='softmax'))
    model.compile(tf.train.GradientDescentOptimizer(lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model.fit(train_samples, train_labels, epochs=40, verbose=0).history

if __name__ == '__main__':
    learning_rates = 10**np.linspace(-1, -3, num=10)
    with ProcessPoolExecutor(1) as executor:
        task_lists = [executor.map(run, product(range(10), [lr]))
                      for lr in learning_rates]
        hist_dfs = [pd.concat([pd.DataFrame(task) for task in tqdm(task_list)])
                    for task_list in tqdm(task_lists)]
    #task_lists = [[run(args) for args in product(range(10), [lr])]
    #              for lr in learning_rates]
    for hist_df in hist_dfs:
        hist_df.reset_index(inplace=True)

    for hist_df, lr in zip(hist_dfs, learning_rates):
        df_mean = hist_df.groupby('index').mean()
        df_std = hist_df.groupby('index').std()
        plt.plot(range(40), df_mean['acc'], label=lr)
        plt.fill_between(range(40), df_mean['acc'] - df_std['acc'],
                         df_mean['acc'] + df_std['acc'], alpha=0.1)
    plt.title('Accuracy for Different Learning Rates')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('linreg_different_LR_ACC.png')
    plt.show()
    for hist_df, lr in zip(hist_dfs, learning_rates):
        df_mean = hist_df.groupby('index').mean()
        df_std = hist_df.groupby('index').std()
        plt.plot(range(40), df_mean['loss'], label=lr)
        plt.fill_between(range(40), df_mean['loss'] - df_std['loss'],
                         df_mean['loss'] + df_std['loss'], alpha=0.1)
    plt.title('Loss for Different Learning Rates')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig('linreg_different_LR_LOSS.png')
    plt.show()
    for hist_df, lr in zip(hist_dfs, learning_rates):
        hist_df.to_csv('linreg_results_lr_{:f}'.format(lr))
