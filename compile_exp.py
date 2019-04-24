
import os
from glob import glob
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_expr(path):
    '''Load experiment in path.'''
    path = Path(path)
    #monitor_name, seed = path.name.rsplit('-', 1)
    alg, learning_rate, gamma, seed = path.name.split('-')
    learning_rate = float(learning_rate)
    gamma = float(gamma)
    seed = int(seed)
    #alg, _, learning_rate, _, gamma = monitor_name.split('-')
    results_list = [pd.read_csv(mname, skiprows=1).assign(seed=seed + i)
                    for i, mname in enumerate(path.glob('*monitor.csv'))]
    #results_list = []
    #for i, monitor_name in enumerate(path.glob('*monitor.csv')):
    #    results = pd.read_csv(monitor_name, skiprows=1)
    #    results['seed'] = seed + i
    #    results_list.append(results)
    results = pd.concat(results_list).assign(alg=alg, gamma=gamma,
                                             learning_rate=learning_rate)
    #results['alg'] = alg
    #results['learning_rate'] = learning_rate
    #results['gamma'] = gamma
    return results

def load_exprs(path):
    '''Load all experiments under path.'''
    path = Path(path)
    expr_paths = [ename for ename in path.iterdir() if ename.is_dir()]
    with ProcessPoolExecutor() as executor:
        expr_dfs = [t for t in tqdm(executor.map(load_expr, expr_paths),
                                    total=len(expr_paths))]
        #expr_dfs = executor.map(load_expr, expr_paths)
    return pd.concat(expr_dfs), expr_paths


def movingAverage(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


if __name__ == '__main__':
    import argparse
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("path", help="The path to the monitor files.")
    PARSER.add_argument("name", help="Name of the compiled pickle file.")
    ARGS = PARSER.parse_args()
    PATH = Path(ARGS.path)
    #expr_dict = get_experiments(PATH)
    #print(expr_dict)
    if PATH.is_dir():
        print('Loading...')
        expr_df, experiment_names = load_exprs(PATH)
        expr_df.reset_index(inplace=True)
        print('Saving...')
        expr_df.to_pickle('{}.pkl'.format(ARGS.name))
