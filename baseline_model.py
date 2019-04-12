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


def discretize(x):
    x = x.ravel()
    x_unique = np.unique(x)
    return np.argmax(x[:, None] == x_unique[None, :], axis=1)

def run(args):
    seed, lr = args
    tf.set_random_seed(seed)
    DATA = np.loadtxt('iris.data', dtype=np.str, delimiter=',')
    SAMPLES, LABELS = np.hsplit(DATA, [-1])
    IMAGE_VECTOR = SAMPLES.shape[-1:]
    LABELS = discretize(LABELS)

    TRAIN_SAMPLES = SAMPLES.astype(np.float32)
    TRAIN_LABELS = LABELS
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(3, input_shape=IMAGE_VECTOR,
                                    activation='softmax'))
    model.compile(tf.train.GradientDescentOptimizer(lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model.fit(TRAIN_SAMPLES, TRAIN_LABELS, epochs=40, verbose=0).history

if __name__ == '__main__':
    learning_rates = [1e-1, 1e-2, 1e-3]
    with ProcessPoolExecutor() as executor:
        task_lists = [executor.map(run, product(range(10), [lr]))
                      for lr in learning_rates]
        hist_dfs = [pd.concat([pd.DataFrame(task) for task in task_list])
                    for task_list in task_lists]
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
