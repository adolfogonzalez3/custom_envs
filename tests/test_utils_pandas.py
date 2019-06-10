
import random
from itertools import chain, product

import gym
import pytest
import numpy as np
import pandas as pd

import custom_envs.utils.utils_pandas as utils

ROW_SIZES = tuple(range(20, 60, 20))
GROUP_SIZES = tuple(range(3, 9, 3))

def create_dataframe(num_of_rows=20):
    data = {chr(97 + i): np.arange(num_of_rows) % (i + 1)
            for i in range(26)}
    dataframe = pd.DataFrame.from_dict(data)
    return dataframe

@pytest.mark.parametrize("num_of_rows", ROW_SIZES)
def test_get_unique(num_of_rows):
    dataframe = create_dataframe(num_of_rows)
    a, c, z= utils.get_unique(dataframe, 'a', 'c', 'z')
    assert len(a) == 1
    assert len(c) == 3
    if num_of_rows < 26:
        assert len(z) == 20
    else:
        assert len(z) == 26

@pytest.mark.parametrize("group_size", GROUP_SIZES[:1])
def test_iterate_levels(group_size):
    dataframe = create_dataframe()
    print(list(dataframe.columns))
    group = random.sample(list(dataframe.columns), group_size)
    print(group)
    dataframe = dataframe.groupby(group).mean()
    print(dataframe)
    idxs_groups = list(utils.iterate_levels(dataframe, group_size))
    idxs, groups = zip(*idxs_groups)
    #print(idxs)
    #print(groups)
    assert len(idxs) == np.prod([ord(g) - 97 for g in group])
    assert sum(len(g) for g in groups) == np.prod([ord(g) - 97 for g in group])


def jtest_create_grid():
    dataframe = create_dataframe().groupby(['a', 'b']).mean()
    (A, B), groups = utils.create_grid(dataframe, levels=2)
    assert A.shape == (10, 10)
    assert B.shape == (10, 10)
    