

from itertools import chain

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

import gym
import pytest
import numpy as np
import pandas as pd

import custom_envs.utils_pandas as utils

def create_dataframe():
    data = [[{'a': i, 'b': j, 'c': 0} for i in range(10)] for j in range(10)]
    data = chain.from_iterable(data)
    return pd.DataFrame(data)

def test_get_unique():
    dataframe = pd.DataFrame(np.arange(25).reshape((5, 5)))
    a, b, c = utils.get_unique(dataframe, 0, 2, 1)
    assert a == [0, 5, 10, 15, 20]
    assert c == [1, 6, 11, 16, 21]
    assert b == [2, 7, 12, 17, 22]


def test_get_unique_1():
    dataframe = create_dataframe()
    a, b = utils.get_unique(dataframe, 'a', 'c')
    assert len(a) == 10
    assert len(b) == 1


def test_iterate_levels():
    dataframe = create_dataframe().groupby(['a', 'b']).mean()
    idxs_groups = list(utils.iterate_levels(dataframe, 2))
    idxs, groups = zip(*idxs_groups)
    assert len(idxs) == 100
    assert sum(len(g) for g in groups) == 100


def test_create_grid():
    dataframe = create_dataframe().groupby(['a', 'b']).mean()
    (A, B), groups = utils.create_grid(dataframe, levels=2)
    assert A.shape == (10, 10)
    assert B.shape == (10, 10)
    



if __name__ == '__main__':
    test_iterate_levels()
