
from itertools import product

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def get_unique(dataframe, *args):
    """
    Returns the unique elements in each column specified.

    :param df: (Pandas DataFrame) The dataframe which contains columns
                                    specified in *args.
    :param *args: ([str]) A list of strings which are a subset of the
                            columns in df.
    :return: ([Pandas Series]) A list of series which hold the unique
                                elements in each column specified.
    """
    return [sorted(dataframe[col].unique()) for col in args]

def iterate_levels(dataframe, levels=3):
    for idx in product(*dataframe.index.levels[:levels]):
        print(idx)
        yield idx, dataframe.loc[idx]

def create_gridv2(dataframe, levels=3):
    idxs_groups = list(iterate_levels(dataframe, levels))
    idxs, groups = zip(*idxs_groups)
    idxs = zip(*idxs)
    grid_shape = [len(l) for l in dataframe.index.levels[:levels]]
    grids = [np.array(idx).reshape(grid_shape) for idx in idxs]
    return grids, groups

def create_grid(df):
    learning_rates = sorted(df['learning_rate'].unique())
    gammas = sorted(df['gamma'].unique())
    algs = sorted(df['alg'].unique())
    X = []
    Y = []
    Z = []
    result_df = df.groupby(['learning_rate', 'gamma', 'alg', 'index']).mean()
    for alg, lr, g in product(algs, learning_rates, gammas):
        result = result_df.loc[(lr, g, alg, pd.IndexSlice['index', :])]
        metric = result['objective']
        X.append(lr)
        Y.append(g)
        Z.append(metric.mean())
    grid_shape = (len(algs), len(learning_rates), len(gammas))
    X = np.array(X, dtype=np.float).reshape(grid_shape)
    Y = np.array(Y, dtype=np.float).reshape(grid_shape)
    Z = np.array(Z, dtype=np.float).reshape(grid_shape)
    return X, Y, Z, result_df

def plot_two(df, df2):
    df['learning_rate'] = np.log10(df['learning_rate'])
    learning_rates = sorted(df['learning_rate'].unique())
    gammas = sorted(df['gamma'].unique())
    algs = sorted(df['alg'].unique())
    result_df = df.groupby(['learning_rate', 'gamma', 'alg', 'index']).mean()
    grids, groups = create_gridv2(result_df)
    Z = np.array([g['objective'].mean()
                  for g in groups]).reshape(grids[0].shape)
    print(Z)
    exit()
    X, Y, Z, result_df = create_grid(df)
    for i in product(*result_df.index.levels[0:3]):
        print(result_df.loc[i])
        print(i)
        exit()
    exit()
    df2['learning_rate'] = np.log10(df['learning_rate'])
    X2, Y2, Z2, result_df2 = create_grid(df2)

    for i, name in enumerate(algs):
        alg = None
        if i == 0:
            alg = 'PPO'
        elif i == 1:
            alg = 'A2C'
        elif i == 2:
            alg = 'DDPG'
        pos = np.nanargmin(Z[i, ...])

        lr = X[i,...].ravel()[pos]
        g = Y[i,...].ravel()[pos]
        result = result_df.loc[(lr, g, alg, pd.IndexSlice['index', :])]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(len(result['r'])), result['objective'])
        print(result)
        print(Z[i])

        lr = X2[i,...].ravel()[pos]
        g = Y2[i,...].ravel()[pos]
        result = result_df2.loc[(lr, g, alg, pd.IndexSlice['index', :])]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(len(result['r'])), result['objective'])
        print(result)
        print(Z[i])
    

def plot_hyperparamsearch_alg(df):
    '''Plot experiments contained in dataframe.'''

    df['learning_rate'] = np.log10(df['learning_rate'])
    learning_rates = sorted(df['learning_rate'].unique())
    gammas = sorted(df['gamma'].unique())
    algs = sorted(df['alg'].unique())
    X, Y, Z, _ = create_grid(df)
    metric_min = np.nanmin(Z)
    metric_max = np.nanmax(Z)

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
        #plt.show()
        #pos = np.nanargmin(Z[i,...])
        #lr = X[i,...].ravel()[pos]
        #g = Y[i,...].ravel()[pos]
        #print(lr, g, alg, pos, X[i,...].size)
        #result = result_df.loc[(lr, g, alg, pd.IndexSlice['index', :])]
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #print(result)
        #print(Z[i])
        #ax.plot(range(len(result['r'])), result['objective'])
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
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("file_name", help="The name of the file to load.")
    ARGS = PARSER.parse_args()
    FILENAME = ARGS.file_name

    print('Reading...')
    df = pd.read_pickle(FILENAME)
    print('Plotting...')
    plot_hyperparamsearch_alg(df)

if __name__ == '__main__':
    print('Reading...')
    df = pd.read_pickle('multiagent.pkl')
    #df2 = pd.read_pickle('single_agent.pkl')
    print('Plotting...')
    #df_lr = pd.read_csv('results_lr.csv', index_col=0)
    #plot_hyperparamsearch_LR()
    plot_two(df, None)