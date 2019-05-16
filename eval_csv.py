
from itertools import product
from collections import defaultdict


import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from custom_envs.utils_pandas import get_unique, create_grid


def plot_two(dataframes, names):
    """
    Plot multiple named dataframes.

    :param dataframes: [pandas.DataFrame] A list of dataframes to plot.
    """
    axes = defaultdict(lambda: plt.figure().add_subplot(111))
    for dataframe, name in zip(dataframes, names):
        dataframe['learning_rate'] = np.log10(dataframe['learning_rate'])
        groups = dataframe.groupby(['learning_rate', 'gamma', 'alg', 'index'])
        mean_df = groups.mean()
        std_df = groups.std()
        (lrs, gammas, algs), groups = create_grid(mean_df)
        metric = np.reshape([g['objective'].mean() for g in groups], lrs.shape)

        algs = [a for a in algs[0, 0, :]]
        for i, alg in enumerate(algs):
            pos = np.nanargmin(metric[..., i])
            lr = lrs[..., i].ravel()[pos]
            g = gammas[..., i].ravel()[pos]
            metric_mean = mean_df.loc[(lr, g, alg), 'objective']
            metric_std = std_df.loc[(lr, g, alg), 'objective']
            print('ALG: ', alg, ' Learning Rate: ', lr, ' Gamma: ', g)
            axes[alg].plot(range(len(metric_mean)), metric_mean, label=name)
            axes[alg].fill_between(range(len(metric_std)),
                                   metric_mean + metric_std,
                                   metric_mean - metric_std, alpha=0.1)
    for (alg, axis) in axes.items():
        axis.set_title(alg)
        axis.set_xlabel('episodes')
        axis.set_ylabel('objective')
        axis.legend()
    plt.show()
    

def plot_hyperparamsearch_alg(dataframe):
    '''Plot experiments contained in dataframe.'''

    dataframe['learning_rate'] = np.log10(dataframe['learning_rate'])
    groups = dataframe.groupby(['learning_rate', 'gamma', 'alg', 'index'])
    mean_df = groups.mean()
    std_df = groups.std()
    (lrs, gammas, algs), groups = create_grid(mean_df)
    Z = np.array([g['objective'].mean() for g in groups]).reshape(lrs.shape)
    metric_min = np.nanmin(Z)
    metric_max = np.nanmax(Z)

    for i, alg in enumerate(algs[0, 0, :]):
        print(alg)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(lrs[..., i], gammas[..., i], Z[..., i], cmap=None,
                        linewidth=0, antialiased=False)
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Gamma')
        ax.set_zlabel('Mean Loss')
        ax.set_zlim(metric_min, metric_max)
        ax.set_title('Model: {}'.format(alg))
        pos = np.nanargmin(Z[..., i])
        lr = lrs[..., i].ravel()[pos]
        g = gammas[..., i].ravel()[pos]
        #print(lr, g, alg, pos, X[i,...].size)
        metric_mean = mean_df.loc[(lr, g, alg), 'objective']
        metric_std = std_df.loc[(lr, g, alg), 'objective']
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(len(metric_mean)), metric_mean)
        ax.fill_between(range(len(metric_std)),
                        metric_mean + metric_std,
                        metric_mean - metric_std, alpha=0.1)
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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="The name of the file to load.")
    args = parser.parse_args()

    print('Reading...')
    dataframe = pd.read_pickle(args.file_name)
    print('Plotting...')
    plot_hyperparamsearch_alg(dataframe)

def main2():
    print('Reading...')
    #dataframe = pd.read_pickle('multiagent.pkl')
    dataframe = pd.read_pickle('single_agent.pkl')
    dataframe2 = pd.read_pickle('exploration_a2c.pkl')
    print('Plotting...')
    #df_lr = pd.read_csv('results_lr.csv', index_col=0)
    #plot_hyperparamsearch_LR()
    plot_two([dataframe, dataframe2], ['single_agent',
                                       'single_agent_with_lower_exploration'])

if __name__ == '__main__':
    main()
