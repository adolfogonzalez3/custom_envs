'''Module for plotting results from experiments.'''
import os
import pathlib
from itertools import product
from collections import defaultdict


import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#import custom_envs.utils.utils_pandas as utils_pandas
import custom_envs.utils.utils_plot as utils_plot
import custom_envs.utils.utils_pandas as utils_pandas
#from eval_multiple_exp import run_experiment_multiagent


def plot(axis_obj, sequence, **kwargs):
    axis_obj.plot(range(len(sequence)), sequence, **kwargs)


def fill_between(axis_obj, mean, std, **kwargs):
    axis_obj.fill_between(range(len(mean)), mean + std, mean - std, **kwargs)


def add_legend(axis):
    chartBox = axis.get_position()
    axis.set_position(
        [chartBox.x0, chartBox.y0, chartBox.width*0.8, chartBox.height])
    axis.legend(loc='upper center', bbox_to_anchor=(
        1.2, 0.8), shadow=True, ncol=1)


def max_group(dataframe, by, column, method='mean'):
    groups = dataframe.groupby(by)
    if method == 'mean':
        return groups.mean()[column].idxmax()


def min_group(dataframe, by, column, method='mean'):
    groups = dataframe.groupby(by)
    if method == 'mean':
        return groups.mean()[column].idxmin()


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
    current_path = pathlib.Path(__file__).resolve().parent
    current_path = current_path / 'results_mnist'

    dataframe['learning_rate'] = np.log10(dataframe['learning_rate'])
    groups = dataframe.groupby(['learning_rate', 'gamma', 'alg', 'index'])
    mean_df = groups.mean()
    std_df = groups.std()
    (lrs, gammas, algs), groups = create_grid(mean_df)
    Z = np.array([g['objective'].mean() for g in groups]).reshape(lrs.shape)
    metric_min = np.nanmin(Z)
    metric_max = np.nanmax(Z)

    rewards_all = []
    accuracies_all = []
    objectives_all = []
    for i, alg in enumerate(algs[0, 0, :]):
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
        metric_mean = mean_df.loc[(lr, g, alg), 'objective']
        metric_std = std_df.loc[(lr, g, alg), 'objective']
        rewards = []
        accuracies = []
        objectives = []
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax1.set_title('{} Rewards'.format(alg))
        ax1.set_ylabel('reward')
        ax1.set_xlabel('step')
        ax2.set_title('{} Accuracies'.format(alg))
        ax2.set_ylabel('accuracy')
        ax2.set_xlabel('step')
        ax3.set_title('{} Loss'.format(alg))
        ax3.set_ylabel('loss')
        ax3.set_xlabel('step')
        for j in range(10):
            rew, acc, obj = run_experiment_multiagent(alg, 10**lr, g, 0,
                                                      current_path)
            rewards.append(rew)
            accuracies.append(acc)
            objectives.append(obj)
            plot(ax1, rew, label='seed {:d}'.format(j))
            plot(ax2, acc, label='seed {:d}'.format(j))
            plot(ax3, obj, label='seed {:d}'.format(j))
        add_legend(ax1)
        add_legend(ax2)
        add_legend(ax3)
        rewards_all.append(rewards)
        accuracies_all.append(accuracies)
        objectives_all.append(objectives)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(len(metric_mean)), metric_mean)
        ax.fill_between(range(len(metric_std)),
                        metric_mean + metric_std,
                        metric_mean - metric_std, alpha=0.1)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax1.set_title('Rewards')
    ax1.set_ylabel('reward')
    ax1.set_xlabel('step')
    ax2.set_title('Accuracies')
    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('step')
    ax3.set_title('Loss')
    ax3.set_ylabel('loss')
    ax3.set_xlabel('step')
    for alg, rewards, accuracies, objectives in zip(algs[0, 0, :], rewards_all, accuracies_all, objectives_all):
        plot(ax1, np.mean(rewards, axis=0), label=alg)
        fill_between(ax1, np.mean(rewards, axis=0), np.std(rewards, axis=0),
                     alpha=0.1)
        plot(ax2, np.mean(accuracies, axis=0), label=alg)
        fill_between(ax2, np.mean(accuracies, axis=0),
                     np.std(accuracies, axis=0), alpha=0.1)
        plot(ax3, np.mean(objectives, axis=0), label=alg)
        fill_between(ax3, np.mean(objectives, axis=0),
                     np.std(objectives, axis=0), alpha=0.1)

    dataframe = pd.read_csv('linreg_results_lr_0.077426.csv').groupby('index')
    mean = dataframe.mean()
    std = dataframe.std()
    plot(ax2, mean['acc'], label='SGD LR=0.077426')
    fill_between(ax2, mean['acc'], std['acc'], alpha=0.1)
    plot(ax3, mean['loss'], label='SGD LR=0.077426')
    fill_between(ax3, mean['loss'], std['loss'], alpha=0.1)

    add_legend(ax1)
    add_legend(ax2)
    add_legend(ax3)
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


def plot_hyperparamsearch_alg2(dataframe):
    '''Plot experiments contained in dataframe.'''

    dataframe['learning_rate'] = np.log10(dataframe['learning_rate'])
    groups = dataframe.groupby(['learning_rate', 'gamma', 'alg', 'index'])
    mean_df = groups.mean()
    std_df = groups.std()
    (lrates, gammas, algs) = utils_pandas.create_grid(mean_df, 3)
    target = np.reshape([g['objective'].mean()
                         for _, g in utils_pandas.iterate_levels(mean_df, 3)],
                        lrates.shape)
    metric_min = np.nanmin(target)
    metric_max = np.nanmax(target)

    for i, alg in enumerate(algs[0, 0, :]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(lrates[..., i], gammas[..., i], target[..., i],
                        cmap=None, linewidth=0, antialiased=False)
        attr = {'xlabel': 'Learning Rate', 'ylabel': 'Gamma',
                'zlabel': 'Mean Loss', 'zmin': metric_min, 'zmax': metric_max,
                'title': 'Model: {}'.format(alg)}
        utils_plot.set_attributes(ax, attr)
        pos = np.nanargmin(target[..., i])
        lrate = lrates[..., i].ravel()[pos]
        gamma = gammas[..., i].ravel()[pos]
        print(alg, 10**lrate, gamma)
        for col in ['objective', 'accuracy']:
            ylabel = 'loss' if col == 'objective' else col
            metric_mean = mean_df.loc[(lrate, gamma, alg), col]
            metric_std = std_df.loc[(lrate, gamma, alg), col]
            fig = plt.figure()
            ax = fig.add_subplot(111)
            attr = {'xlabel': 'steps', 'ylabel': ylabel,
                    'title': 'Model: {}'.format(alg)}
            utils_plot.set_attributes(ax, attr)
            utils_plot.plot_sequence(ax, metric_mean)
            utils_plot.fill_between(ax, metric_mean, metric_std, alpha=0.1)
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", help="The name of the file to load.")
    args = parser.parse_args()

    print('Reading...')
    dataframe = pd.read_pickle(args.file_name)
    print(dataframe)
    print('Plotting...')
    plot_hyperparamsearch_alg2(dataframe)


def main2():
    print('Reading...')
    #dataframe = pd.read_pickle('multiagent.pkl')
    dataframe = pd.read_pickle('single_agent.pkl')
    dataframe2 = pd.read_pickle('exploration_a2c.pkl')
    print('Plotting...')
    #df_lr = pd.read_csv('results_lr.csv', index_col=0)
    # plot_hyperparamsearch_LR()
    plot_two([dataframe, dataframe2], ['single_agent',
                                       'single_agent_with_lower_exploration'])


def main3():
    dataframe = pd.read_csv(
        'linreg_results_lr_0.061585.csv').groupby('index').mean()
    print(dataframe)
    print(dataframe['loss'])
    print(dataframe['acc'])
    print(dataframe['acc'].max())


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main()
