
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_expr(path):
    '''Load experiment in path.'''
    path = Path(path)
    alg, learning_rate, gamma, seed = path.name.split('-')
    learning_rate = float(learning_rate)
    gamma = float(gamma)
    seed = int(seed)
    results_list = [pd.read_csv(mname).assign(seed=seed + i)
                    for i, mname in enumerate(path.glob('*.mon.csv'))]
    results = pd.concat(results_list).assign(alg=alg, gamma=gamma,
                                             learning_rate=learning_rate)
    return results

def load_exprs(path):
    '''Load all experiments under path.'''
    path = Path(path)
    expr_paths = [ename for ename in path.iterdir() if ename.is_dir()]
    with ProcessPoolExecutor() as executor:
        expr_dfs = list(tqdm(executor.map(load_expr, expr_paths, chunksize=32),
                             total=len(expr_paths)))
    return pd.concat(expr_dfs)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="The path to the monitor files.")
    parser.add_argument("name", help="Name of the compiled pickle file.")
    args = parser.parse_args()
    path = Path(args.path)
    #expr_dict = get_experiments(PATH)
    #print(expr_dict)
    if path.is_dir():
        print('Loading...')
        expr_df = load_exprs(path).reset_index()
        print('Saving...')
        expr_df.to_pickle('{}.pkl'.format(args.name))


if __name__ == '__main__':
    main()
