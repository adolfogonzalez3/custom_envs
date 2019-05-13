
import json
import sqlite3
from pathlib import Path
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

def create_json_file(file_name):
    algs = ['A2C', 'PPO']
    learning_rates = 10**np.linspace(-1, -3, 10)
    gammas = 10**np.linspace(0, -1, 10)
    seeds = list(range(20))
    tasks = [json.dumps({'alg': alg, 'learning_rate': lr, 'gamma': g,
                         'seed': s})
             for alg, lr, g, s in product(algs, learning_rates, gammas, seeds)]
    with file_name.open('wt') as js_file:
        js_file.write('\n'.join(tasks))
    

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('file_name')
    parser.add_argument('--type', choices=['json', 'csv', 'sqlite3'],
                        default='json')

    args = parser.parse_args()
    file_name = Path(args.file_name)

    if args.type == 'json':
        create_json_file(file_name.with_suffix('.json'))
    elif args.type == 'csv':
        create_tasks_file(file_name.with_suffix('.csv'))
    elif args.type == 'sqlite3':
        create_database(file_name.with_suffix('.db'))

if __name__ == '__main__':
    main()
