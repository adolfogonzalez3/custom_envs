'''Module for testing utils_pandas.'''
import random

import pytest
import numpy as np
import pandas as pd

import custom_envs.utils.utils_pandas as utils

ROW_SIZES = tuple(range(50, 200, 50))
GROUP_SIZES = tuple(range(2, 4))


def create_dataframe(num_of_rows=50):
    '''Create a dataframe for use in tests.'''
    data = {chr(97 + i): np.arange(num_of_rows) % (i + 1)
            for i in range(26)}
    return pd.DataFrame.from_dict(data)


@pytest.mark.parametrize("num_of_rows", ROW_SIZES)
def test_get_unique(num_of_rows):
    '''Test get_unique.'''
    dataframe = create_dataframe(num_of_rows)
    col_a, col_c, col_z = utils.get_unique(dataframe, 'a', 'c', 'z')
    assert len(col_a) == 1
    assert len(col_c) == 3
    assert len(col_z) == 26


@pytest.mark.parametrize("num_of_rows", ROW_SIZES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
def test_iterate_levels(num_of_rows, group_size):
    '''
    Test iterate levels.

    Randomly sample columns from the dataframe and then get the number of
    unique values for each column and then multiply together to get the
    total number of groups.
    '''
    dataframe = create_dataframe(num_of_rows)
    col_idxs = random.sample(range(len(dataframe.columns)), group_size)
    group_cols = [dataframe.columns[i] for i in col_idxs]
    col_data = utils.get_unique(dataframe, *group_cols)
    group_no = np.prod([len(col) for col in col_data])
    dataframe = dataframe.groupby(group_cols).mean()
    idxs_groups = list(utils.iterate_levels(dataframe, group_size))
    idxs, groups = zip(*idxs_groups)
    assert len(idxs) == group_no
    assert sum(len(g) for g in groups) <= num_of_rows


@pytest.mark.parametrize("num_of_rows", ROW_SIZES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
def test_create_grid(num_of_rows, group_size):
    '''
    Test create_grid.

    Randomly sample columns from the dataframe and then get the number of
    unique values for each column and then multiply together to get the
    total number of groups.
    '''
    dataframe = create_dataframe(num_of_rows)
    col_idxs = random.sample(range(len(dataframe.columns)), group_size)
    group_cols = [dataframe.columns[i] for i in col_idxs]
    col_data = utils.get_unique(dataframe, *group_cols)
    grid_shape = tuple(len(col) for col in col_data)
    dataframe = dataframe.groupby(group_cols).mean()
    grids = utils.create_grid(dataframe, levels=group_size)
    for grid in grids:
        assert grid.shape == grid_shape


@pytest.mark.parametrize("num_of_rows", ROW_SIZES)
def test_apply_method(num_of_rows):
    '''Test apply_method.'''
    dataframe = create_dataframe(num_of_rows)
    dataframe_mean = utils.apply_method(dataframe, method='mean')
    assert (dataframe_mean == np.mean(dataframe, axis=0)).all()


@pytest.mark.parametrize("num_of_rows", ROW_SIZES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
def test_max_group(num_of_rows, group_size):
    '''Test max_group.'''
    dataframe = create_dataframe(num_of_rows)
    col_idxs = random.sample(range(len(dataframe.columns)), group_size+1)
    *group_cols, target = [dataframe.columns[i] for i in col_idxs]
    index = utils.max_group(dataframe, group_cols, target, method='mean')
    dataframe_mean = dataframe.groupby(group_cols).mean()
    idxs_groups = list(utils.iterate_levels(dataframe_mean, group_size))
    idxs, groups = zip(*idxs_groups)
    max_value = max(g[target].mean() for g in groups)
    assert index in idxs
    assert dataframe_mean.loc[index, target] == max_value


@pytest.mark.parametrize("num_of_rows", ROW_SIZES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
def test_min_group(num_of_rows, group_size):
    '''Test min_group.'''
    dataframe = create_dataframe(num_of_rows)
    col_idxs = random.sample(range(len(dataframe.columns)), group_size+1)
    *group_cols, target = [dataframe.columns[i] for i in col_idxs]
    index = utils.min_group(dataframe, group_cols, target, method='mean')
    dataframe_mean = dataframe.groupby(group_cols).mean()
    idxs_groups = list(utils.iterate_levels(dataframe_mean, group_size))
    idxs, groups = zip(*idxs_groups)
    max_value = min(g[target].mean() for g in groups)
    assert index in idxs
    assert dataframe_mean.loc[index, target] == max_value
