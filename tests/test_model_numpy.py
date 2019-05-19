
import pytest
import numpy as np
import numpy.random as npr

from custom_envs.models.model_numpy import ModelNumpy as Model


def test_size():
    model = Model(4, 3)
    assert model.size == 12

def test_forward():
    model = Model(4, 3)
    features = np.ones((32, 4))
    a = model.forward(features)
    assert a.shape == (32, 3)

def test_compute_loss():
    model = Model(4, 3)
    features = np.ones((32, 4))
    labels = npr.randint(3, size=(32, 3))
    loss = model.compute_loss(features, labels)
    assert isinstance(loss, float)

def test_compute_gradients():
    model = Model(4, 3)
    features = np.ones((32, 4))
    labels = npr.randint(3, size=(32, 3))
    grad = model.compute_gradients(features, labels)
    assert grad.shape == (4, 3)

def test_compute_accuracy():
    model = Model(4, 3)
    features = np.ones((32, 4))
    labels = npr.randint(3, size=(32, 3))
    acc = model.compute_loss(features, labels)
    assert isinstance(acc, float)

def test_compute_backprop():
    model = Model(4, 3)
    features = np.ones((32, 4))
    labels = npr.randint(3, size=(32, 3))
    loss, grad, acc = model.compute_backprop(features, labels)
    assert isinstance(loss, float)
    assert grad.shape == (4, 3)
    assert isinstance(acc, float)
