'''Utilities for manipulating pandas DataFrames.'''
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

    :param dataframe: (pandas.DataFrame) A dataframe containing a tiered index.
    :param levels: (int) The number of levels to index by.
    """
    indexes_groups = list(iterate_levels(dataframe, levels))
    indexes, groups = zip(*indexes_groups)
    indexes = zip(*indexes)
    grid_shape = [len(l) for l in dataframe.index.levels[:levels]]
    grids = [np.reshape(index, grid_shape) for index in indexes]
    return grids, groups


def apply_method(dataframe, method):
    """
    Apply method to dataframe.

    :param dataframe: (pandas.DataFrame) The dataframe to apply to the method.
    :param method: (str) The method to apply to the method.
    """
    if method == 'mean':
        return dataframe.mean()
    else:
        raise RuntimeError('No method named: {}'.format(method))


def max_group(dataframe, groupby, column, method='mean'):
    """
    Get the group with max value after applying a method.

    :param dataframe: (pandas.DataFrame) The dataframe to use.
    :param by: ([str]) A list of columns to group by.
    :param column: (str) The column to target.
    :param method: (str) The method to apply to the dataframe.
    """
    groups = dataframe.groupby(groupby)
    group_applied = apply_method(groups, method)
    return group_applied[column].idmax()


def min_group(dataframe, groupby, column, method='mean'):
    """
    Get the group with min value after applying a method.

    :param dataframe: (pandas.DataFrame) The dataframe to use.
    :param by: ([str]) A list of columns to group by.
    :param column: (str) The column to target.
    :param method: (str) The method to apply to the dataframe.
    """
    groups = dataframe.groupby(groupby)
    groups_applied = apply_method(groups, method)
    return groups_applied[column].idxmin()
