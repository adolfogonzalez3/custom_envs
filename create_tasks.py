'''
Module for creating files that are contain hyperparameters for experiments.
'''
import json
import sqlite3
import argparse
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd


def populate_table(hyperparams):
    '''
    Populate a table with all combinations of hyperparams.

    :param hyperparams: (dict) A dictionary of lists where the key is the name
                               of the hyperparameter and the associated value
                               is a list of possible values for that parameter.
    :return: (pandas.DataFrame) A dataframe containing all possible
                                combinations of hyperparameters.
    '''
    tasks = [{name: param for name, param in zip(hyperparams.keys(), params)}
             for params in product(*list(hyperparams.values()))]
    return pd.DataFrame(tasks)


def main():
    '''Create a file containing experiment details.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name')
    parser.add_argument('--type', choices=['json', 'csv', 'sqlite3'],
                        default='csv')

    args = parser.parse_args()
    if args.type == 'sqlite3':
        suffix = '.db'
    else:
        suffix = '.' + args.type

    file_name = Path(args.file_name).with_suffix(suffix)

    columns = ['alg', 'learning_rate', 'gamma', 'seed']
    hyperparams = {}
    hyperparams['seed'] = list(range(3))
    hyperparams['gamma'] = 10**np.linspace(0, -1, 5)
    hyperparams['learning_rate'] = 10**np.linspace(-2, -4, 5)
    hyperparams['alg'] = ['A2C', 'PPO']

    dataframe = populate_table(hyperparams)
    dataframe = dataframe.reindex(columns=columns)
    dataframe['total_timesteps'] = 10**7
    dataframe['env_name'] = 'MultiOptimize-v0'
    dataframe['path'] = 'results_mnist_multioptimize'
    kwargs = {'data_set': 'iris', 'batch_size': 32}
    dataframe['kwargs'] = json.dumps(kwargs)

    if args.type == 'json':
        dataframe.to_json(file_name, orient='records', lines=True)
    elif args.type == 'csv':
        dataframe.to_csv(file_name, header=False, index=False,)
    elif args.type == 'sqlite3':
        dataframe['done'] = False
        with sqlite3.connect(str(file_name)) as conn:
            dataframe.to_sql('hyperparameters', conn)


if __name__ == '__main__':
    main()
