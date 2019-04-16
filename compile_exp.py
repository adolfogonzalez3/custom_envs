
import os
from glob import glob
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_expr(path):
    '''Load experiment in path.'''
    path = Path(path)
    #print(path.name.rsplit('-', 1))
    monitor_name, seed = path.name.rsplit('-', 1)
    alg, _, learning_rate, _, gamma = monitor_name.split('_')
    results = pd.read_csv(path / 'monitor.csv', skiprows=1)
    results['alg'] = alg
    results['learning_rate'] = learning_rate
    results['gamma'] = gamma
    results['seed'] = seed
    return results

def load_exprs(path):
    '''Load all experiments under path.'''
    path = Path(path)
    exper_names = os.listdir(path)
    expr_paths = [path / ename for ename in exper_names]
    with ThreadPoolExecutor() as executor:
        expr_dfs = executor.map(load_expr, expr_paths)
    return pd.concat(expr_dfs), exper_names


def movingAverage(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results2(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    #y = movingAverage(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

def contract(path):
    files = glob(os.path.join(path, '**', '*.csv'), recursive=True)
    #files = [stuff for stuff in os.walk(path)]
    filepaths = sorted(files)
    projects = defaultdict(list)
    for filepath in filepaths:
        basename = os.path.dirname(filepath).rsplit('-', 1)[0]
        projects[basename].append(filepath)
    
    return projects

def plot_each_expr(experiments, experiment_names):
    for expr_df, expr_name in zip(experiments, experiment_names):
        expr_df_grouped = expr_df.groupby('index').mean()
        mean = (expr_df_grouped['r']).rolling(100).mean()
        std = (expr_df_grouped['r']).rolling(100).std()
        plt.plot(expr_df_grouped['l'].cumsum(), mean, label=expr_name)
        plt.fill_between(expr_df_grouped['l'].cumsum(), mean - std, mean+std,
                         alpha=0.1)
        #plt.title(env_name)
    plt.title('All Envs')
    plt.ylabel('Reward')
    #plt.ylabel('Accuracy')
    #plt.ylim(0,1)
    plt.xlabel('time step')
    plt.legend()
    plt.show()

def plot_each_expr_3d(experiments, experiment_names):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    learning_rate_list = []
    gamma_list = []
    steps = []
    for expr_df, expr_name in zip(experiments, experiment_names):
        expr_df_grouped = expr_df.groupby('index').mean()
        expr_df_grouped['l'] = expr_df_grouped['l'].cumsum()
        #steps.append(expr_df_grouped['objective'].min())
        steps.append(expr_df_grouped[expr_df_grouped['accuracy'] > 0.9]['l'].min())
        learning_rate, gamma = expr_name.split('_')
        learning_rate = np.log10(float(learning_rate.strip('lr')))
        gamma = float('.'+gamma.strip('g'))
        learning_rate_list.append(learning_rate)
        gamma_list.append(gamma)
    #plt.title(env_name)
    learning_rate_list = np.array(learning_rate_list)
    gamma_list = np.array(gamma_list)
    steps = np.array(steps, dtype=np.float)
    steps[np.isnan(steps)] = np.max(steps[np.logical_not(np.isnan(steps))])
    X = learning_rate_list.reshape((3, 3)).T
    Y = gamma_list.reshape((3, 3)).T
    Z = steps.reshape((3, 3)).T
    print(learning_rate_list, gamma_list, steps)
    print(X, Y, Z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(learning_rate_list, gamma_list, steps)
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('Learning Rate Log10')
    ax.set_ylabel('Gamma')
    ax.set_zlabel('Number of Time Steps until Accuracy > 0.90')
    ax.set_title('Number of Time Steps until Accuracy > 0.90.')
    plt.show()

def plot_expr(experiment, experiment_name):
    max_index = experiment['index'].max()
    experiment = experiment.sort_values('index')
    for i in range(10):
        expr_df_grouped = experiment[i:max_index*10:10]
        mean = (expr_df_grouped['accuracy']).rolling(100).mean()
        std = (expr_df_grouped['accuracy']).rolling(100).std()
        plt.plot(expr_df_grouped['l'].cumsum(), mean, label=i)
        plt.fill_between(expr_df_grouped['l'].cumsum(), mean - std, mean+std,
                         alpha=0.1)
    plt.title('All Seeds for learning rate = 1e-3 and gamma = 0.6336.')
    plt.ylabel('Loss')
    #plt.ylabel('Accuracy')
    #plt.ylim(0,1)
    plt.xlabel('time step')
    plt.legend()
    plt.show()

def get_experiments(path):
    experiments = os.listdir(path)
    expr_names = [e.rsplit('-', 1)[0] for e in experiments]
    return Counter(expr_names)

if __name__ == '__main__':
    import argparse
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("path", help="The path to the files.")
    ARGS = PARSER.parse_args()
    PATH = Path(ARGS.path)
    #expr_dict = get_experiments(PATH)
    #print(expr_dict)
    if PATH.is_dir():
        print('Loading...')
        expr_df, experiment_names = load_exprs(PATH)
        expr_df.reset_index(inplace=True)
        print('Saving...')
        expr_df.to_pickle('{}.pkl'.format(PATH.name))
