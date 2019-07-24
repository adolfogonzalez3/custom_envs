'''A script to plot data from dataframes.'''
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

import custom_envs.utils.utils_plot as utils_plot


def plot_results(axes, dataframe, groupby, label=None):
    '''Plot results on multiple axes given a dataframe.'''
    groupby = {groupby} if isinstance(groupby, str) else set(groupby)
    grouped = dataframe.groupby(groupby)
    mean_df = grouped.mean()
    std_df = grouped.std()
    columns = set(mean_df.columns) & set(axes.keys()) - groupby
    for name in columns:
        utils_plot.plot_sequence(axes[name], mean_df[name], label=label)
        utils_plot.fill_between(axes[name], mean_df[name], std_df[name],
                                alpha=0.1, label=label)


def main():
    '''Evaluate a trained model against logistic regression.'''
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="The path to the file.",
                        type=Path)
    parser.add_argument('--rolling', help='Set a rolling window.',
                        type=int, default=0)
    args = parser.parse_args()
    dataframe = pd.read_csv(args.path)
    axes = defaultdict(lambda: plt.figure().add_subplot(111))
    columns = dataframe.columns
    pyplot_attr = {
        'title': 'Performance on {} data set',
        'xlabel': 'Epoch',
    }
    dataframe = dataframe.fillna(method='bfill')
    if args.rolling != 0:
        dataframe = dataframe.rolling(args.rolling).mean()
    for name in columns:
        utils_plot.plot_sequence(axes[name], dataframe[name], label=name)
    for column in columns:
        pyplot_attr['ylabel'] = column.capitalize()
        utils_plot.set_attributes(axes[column], pyplot_attr)
    plt.show()


if __name__ == '__main__':
    main()
