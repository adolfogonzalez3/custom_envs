
from pathlib import Path

import numpy as np
import pandas as pd

def verify_csv(exp_path, csv_path):
    exp_path = Path(exp_path)
    csv_path = Path(csv_path)
    completed_exps = {path.name for path in exp_path.iterdir()
                      if path.is_dir()}
    incomplete_exps = []
    with csv_path.open() as csv:
        for line in csv:
            task = line.rstrip()
            alg, lr, g, seed, *_  = task.split(',')
            lr = float(lr)
            g = float(g)
            seed = int(seed)
            file_name = '{}-{:.4f}-{:.4f}-{:d}'.format(alg, lr, g, seed)
            if file_name not in completed_exps:
                incomplete_exps.append((alg, lr, g, seed))
    incomplete_exprs = sorted(incomplete_exps, key=lambda x: x[-1:1:-1])
    print(len(incomplete_exprs))
    print(incomplete_exprs)
    return incomplete_exprs

def write_log(exprs):
    pass

def main():
    import argparse
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("experiments_path", help=("The path to the "
                                                  "experiments folders."))
    PARSER.add_argument("log_path", help=("The path to the log which contains "
                                          "all experiments."))
    PARSER.add_argument("save_path", help=("The path to save the experiments "
                                           "not completed."))
    ARGS = PARSER.parse_args()
    incomplete_exprs = verify_csv(ARGS.experiments_path, ARGS.log_path)
    with open(ARGS.log_path, 'rt') as log:
        *_, env, path = log.readline().rstrip().split(',')
    with open(ARGS.save_path, 'wt') as csv:
        for args in incomplete_exprs:
            csv.write(','.join(args + (env, path)) + '\n')

if __name__ == '__main__':
    main()
