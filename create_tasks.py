
import sqlite3
from itertools import product

import numpy as np

def create_tasks_file(file_name):
    algs = ['A2C', 'PPO']
    learning_rate = [str(i) for i in 10**np.linspace(-1, -3, 10)]
    gamma = [str(i) for i in 10**np.linspace(0, -1, 10)]
    seed = list(str(i) for i in range(20))
    with open(file_name, 'wt') as csv:
        for args in product(algs, learning_rate, gamma, seed):
            csv.write(','.join(args + ('Optimize-v0', 'results')) + '\n')

def create_database(file_name):
    algs = ['A2C', 'DDPG', 'PPO']
    learning_rates = 10**np.linspace(-1, -3, 10)
    gamma = 10**np.linspace(0, -1, 10)
    seed = list(range(20))
    with sqlite3.connect(file_name) as conn:
        with conn:
            conn.execute(('create table experiments (alg text, lr real, '
                          'gamma real, seed integer, done integer)'))
            
            conn.executemany('insert into experiments values (?,?,?,?,?)',
                             product(algs, learning_rates, gamma, seed, [0]))

if __name__ == '__main__':
    import argparse

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('file_name')
    PARSER.add_argument('--type', choices=['csv', 'sqlite3'], default='csv')

    ARGS = PARSER.parse_args()

    if ARGS.type == 'csv':
        create_tasks_file(ARGS.file_name)
    elif ARGS.type == 'sqlite3':
        create_database(ARGS.file_name)
