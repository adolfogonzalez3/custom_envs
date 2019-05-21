
from itertools import product

import numpy as np


def get_unique(dataframe, *args):
    """
    Return the unique elements in each column specified.

    :param df: (Pandas DataFrame) The dataframe which contains columns
                                    specified in *args.
    :param *args: ([str]) A list of strings which are a subset of the
                            columns in df.
    :return: ([Pandas Series]) A list of series which hold the unique
                                elements in each column specified.
    """
    return [sorted(dataframe[col].unique()) for col in args]


def iterate_levels(dataframe, levels=3):
    """
    Yield the levels in the dataframe as well as their associated rows.

    :param dataframe: (pandas.DataFrame) A dataframe contained a tiered index
                                         containing at least (levels) levels.
    :param levels: (int) The number of levels to index by.
    """
    for idx in product(*dataframe.index.levels[:levels]):
        yield idx, dataframe.loc[idx:idx]


def create_grid(dataframe, levels=3):
    """
    Create grids and groups of rows based on a tiered index.

    :param dataframe: (pandas.DataFrame) A dataframe contained a tiered index
                                         containing at least (levels) levels.
    :param levels: (int) The number of levels to index by.
    """
    idxs_groups = list(iterate_levels(dataframe, levels))
    idxs, groups = zip(*idxs_groups)
    idxs = zip(*idxs)
    grid_shape = [len(l) for l in dataframe.index.levels[:levels]]
    grids = [np.array(idx).reshape(grid_shape) for idx in idxs]
    return grids, groups

def max_group(dataframe, by, column, method='mean'):
    groups = dataframe.groupby(by)
    if method == 'mean':
        return groups.mean()[column].idxmax()

def min_group(dataframe, by, column, method='mean'):
    groups = dataframe.groupby(by)
    if method == 'mean':
        return groups.mean()[column].idxmin()