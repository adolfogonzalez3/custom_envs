
from itertools import product

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot_hyperparamsearch_alg(df):
    '''Plot experiments contained in dataframe.'''

    learning_rates = sorted(df['learning_rate'].unique())
    gammas = sorted(df['gamma'].unique())
    algs = sorted(df['alg'].unique())
    X = []
    Y = []
    Z = []
    test = []
    result_df = df.groupby(['learning_rate', 'gamma', 'alg', 'index']).mean()
    for alg, lr, g in product(algs, learning_rates, gammas):
        result = result_df.loc[(lr, g, alg, pd.IndexSlice['index', :])]
        metric = result['objective']
        test.append(alg)
        X.append(lr)
        Y.append(g)
        Z.append(metric.mean())
    
    grid_shape = (len(algs), len(learning_rates), len(gammas))
    X = np.array(X, dtype=np.float).reshape(grid_shape)
    Y = np.array(Y, dtype=np.float).reshape(grid_shape)
    Z = np.array(Z, dtype=np.float).reshape(grid_shape)
    metric_min = np.min(Z)
    metric_max = np.max(Z)

    for i, name in enumerate(algs):
        alg = None
        if i == 0:
            alg = 'PPO'
        elif i == 1:
            alg = 'A2C'
        elif i == 2:
            alg = 'DDPG'
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #ax.scatter(learning_rate_list, gamma_list, steps)
        ax.plot_surface(X[i], Y[i], Z[i], cmap=None, linewidth=0,
                        antialiased=False)
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Gamma')
        ax.set_zlabel('Mean Loss')
        ax.set_zlim(metric_min, metric_max)
        ax.set_title('Model: {}'.format(alg))
        plt.show()

def plot_hyperparamsearch_LR():
    df_lr = pd.read_csv('results_lr.csv', index_col=0)

    learning_rates_lr = sorted(df_lr['learning_rate'].unique())

    for lr in learning_rates_lr:
        result = df_lr.query('learning_rate == @lr')
        result = result.groupby('index').mean()
        metric = result['loss']
        plt.plot(range(len(metric)), metric, label=str(lr))
    plt.legend()
    plt.title('Comparison of Linear Regression Learning Rates')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

def plot_best(df, df_lr):
    learning_rates_lr = sorted(df_lr['learning_rate'].unique())

    best_lr = None
    best_metric_lr = np.inf
    for lr in learning_rates_lr:
        result = df_lr.query('learning_rate == @lr')
        result = result.groupby('index').mean()
        metric = result['loss']
        if metric.min() < best_metric_lr:
            best_metric_lr = metric.min()
            best_lr = metric

    learning_rates = sorted(df['learning_rate'].unique())
    gammas = sorted(df['gamma'].unique())
    algs = sorted(df['alg'].unique())

    best = [None]*len(algs)
    best_metric = [np.inf]*len(algs)
    for alg, lr, g in product(algs, learning_rates, gammas):
        result = df.query('learning_rate == @lr & gamma == @g & alg == @alg')
        result = result.groupby('index').mean()
        metric = result['objective']
        if metric.min() < best_metric[alg]:
            best_metric[alg] = metric.min()
            best[alg] = metric
    
    for metric, i in zip(best, algs):
        alg = None
        if i == 0:
            alg = 'PPO'
        elif i == 1:
            alg = 'A2C'
        elif i == 2:
            alg = 'DDPG'
        plt.plot(range(len(metric)), metric, label=alg)
    plt.plot(range(len(best_lr)), best_lr, label='Linear Regression')
    plt.legend()
    plt.title('Comparison of PPO vs A2C vs DDPG vs Linear Regression')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


if __name__ == '__main__':
    import argparse
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("file_name", help="The name of the file to load.")
    ARGS = PARSER.parse_args()
    FILENAME = ARGS.file_name

    print('Reading...')
    df = pd.read_pickle(FILENAME)
    print('Plotting...')
    plot_hyperparamsearch_alg(df)
    #df_lr = pd.read_csv('results_lr.csv', index_col=0)
    #plot_hyperparamsearch_LR()
