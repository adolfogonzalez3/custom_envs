
import pytest
import numpy as np
import numpy.random as npr

from custom_envs.models import ModelTF, ModelKeras, ModelNumpy

MODELS = [ModelNumpy, ModelKeras, ModelTF]


@pytest.mark.parametrize("Model", MODELS)
def test_size(Model):
    model = Model(4, 3)
    assert model.size == 12


@pytest.mark.parametrize("Model", MODELS)
def test_forward(Model):
    model = Model(4, 3)
    features = np.ones((32, 4))
    a = model.forward(features)
    assert a.shape == (32, 3)


@pytest.mark.parametrize("Model", MODELS)
def test_compute_loss(Model):
    model = Model(4, 3)
    features = np.ones((32, 4))
    labels = npr.randint(3, size=(32, 3))
    loss = model.compute_loss(features, labels)
    assert isinstance(loss, float)


@pytest.mark.parametrize("Model", MODELS)
def test_compute_gradients(Model):
    model = Model(4, 3)
    features = np.ones((32, 4))
    labels = npr.randint(3, size=(32, 3))
    grad = model.compute_gradients(features, labels)
    assert grad.shape == (4, 3)


@pytest.mark.parametrize("Model", MODELS)
def test_compute_accuracy(Model):
    model = Model(4, 3)
    features = np.ones((32, 4))
    labels = npr.randint(3, size=(32, 3))
    acc = model.compute_loss(features, labels)
    assert isinstance(acc, float)


@pytest.mark.parametrize("Model", MODELS)
def test_compute_backprop(Model):
    model = Model(4, 3)
    features = np.ones((32, 4))
    labels = npr.randint(3, size=(32, 3))
    loss, grad, acc = model.compute_backprop(features, labels)
    assert isinstance(loss, float)
    assert grad.shape == (4, 3)
    assert isinstance(acc, float)
