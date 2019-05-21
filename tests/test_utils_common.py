
from itertools import chain

import gym
import pytest
import numpy as np
import pandas as pd
import numpy.random as npr

import custom_envs.utils.utils_common as utils
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv


def test_shuffle():
    A = np.arange(25, 0, -1)
    B = np.arange(25, 0, -1)
    a_shuffle, b_shuffle = utils.shuffle(A, B)
    assert np.all(a_shuffle == b_shuffle)


def test_range_slice():
    A = np.arange(256)
    for i, batch in enumerate(utils.range_slice(0, 256, 32)):
        assert np.all(A[batch] == A[i*32:(i+1)*32])


def test_batchify_zip():
    A = np.arange(256)
    B = np.arange(256, 512)
    for i, (a, b) in enumerate(utils.batchify_zip(A, B, size=32)):
        assert len(a) == 32
        assert len(b) == 32
        assert np.all(a == A[i*32:32*(i+1)])
        assert np.all(b == B[i*32:32*(i+1)])
        assert np.all(256+a == b)


def test_cross_entropy():
    p = utils.softmax(npr.rand(1, 10))
    y = np.zeros((1, 10))
    y[:, 5] = 1
    cost = utils.cross_entropy(p, y)
    assert cost >= 0
    assert isinstance(cost, float)


def test_cross_entropy_batch():
    p = utils.softmax(npr.rand(10, 10))
    y = np.zeros((10, 10))
    y[:, 5] = 1
    cost = utils.cross_entropy(p, y)
    assert cost >= 0
    assert isinstance(cost, float)


def test_mse():
    p = npr.rand(1, 10)
    p = p - np.min(p) / np.ptp(p)
    y = np.zeros((1, 10))
    y[:, 5] = 1
    cost = utils.mse(p, y)
    assert cost >= 0
    assert isinstance(cost, float)


def test_mse_batch():
    p = npr.rand(10, 10)
    p = p - np.min(p) / np.ptp(p)
    y = np.zeros((10, 10))
    y[:, 5] = 1
    cost = utils.mse(p, y)
    assert cost >= 0
    assert isinstance(cost, float)


def test_softmax():
    p = npr.randint(-10, 10, size=(10, 5))
    p = utils.softmax(p)
    assert np.all(p >= 0)
    assert np.all(p <= 1)
    assert np.all(np.isclose(np.sum(p, axis=1), 1))


def test_softmax_large():
    p = npr.randint(-1e9, 1e9, size=(10, 5))
    p = utils.softmax(p)
    assert np.all(p >= 0)
    assert np.all(p <= 1)
    assert np.all(np.sum(p, axis=1) == 1)


def test_sigmoid():
    p = npr.randint(-10, 10, size=(10, 5))
    p = utils.sigmoid(p)
    assert np.all(p >= 0)
    assert np.all(p <= 1)


def test_sigmoid_large():
    p = npr.randint(-1e9, 1e9, size=(10, 5))
    p = utils.sigmoid(p)
    assert np.all(p >= 0)
    assert np.all(p <= 1)


def test_to_onehot():
    A = [i % 5 + 10 for i in range(10)]
    B, num_of_labels = utils.to_onehot(A)
    assert len(np.unique(A)) == 5
    assert num_of_labels == 5
    assert np.all(B[:5] == B[5:])
    assert num_of_labels == 5


def test_normalize():
    A = npr.randn(5, 5)
    A = utils.normalize(A)
    assert np.all(A >= 0)
    assert np.all(A <= 1)
