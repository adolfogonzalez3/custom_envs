
import pytest
import numpy as np
import numpy.random as npr

from custom_envs.models import ModelNumpy, ModelKeras, ModelTF

MODELS = [ModelNumpy, ModelKeras, ModelTF]


@pytest.mark.parametrize("Model", MODELS)
def test_size(Model):
    model = Model(4, 3)
    assert model.size == 12


@pytest.mark.parametrize("Model", MODELS)
def test_forward_shape(Model):
    model = Model(4, 3)
    features = np.ones((32, 4))
    a = model.forward(features)
    assert a.shape == (32, 3)


@pytest.mark.parametrize("Model", MODELS)
def test_compute_loss_type(Model):
    model = Model(4, 3)
    features = np.ones((32, 4))
    labels = npr.randint(3, size=(32, 3))
    loss = model.compute_loss(features, labels)
    assert isinstance(loss, float)


@pytest.mark.parametrize("Model", MODELS)
def test_compute_gradients_shape(Model):
    model = Model(4, 3)
    features = np.ones((32, 4))
    labels = npr.randint(3, size=(32, 3))
    grad = model.compute_gradients(features, labels)
    assert grad.shape == (4, 3)


@pytest.mark.parametrize("Model", MODELS)
def test_compute_accuracy_type_bound(Model):
    model = Model(4, 3)
    features = np.ones((32, 4))
    labels = npr.randint(3, size=(32, 3))
    acc = model.compute_accuracy(features, labels)
    assert isinstance(acc, float)
    assert 0 <= acc <= 1


@pytest.mark.parametrize("Model", MODELS)
def test_compute_backprop_type_bound_shape(Model):
    model = Model(4, 3)
    features = np.ones((32, 4))
    labels = npr.randint(3, size=(32, 3))
    loss, grad, acc = model.compute_backprop(features, labels)
    assert isinstance(loss, float)
    assert grad.shape == (4, 3)
    assert isinstance(acc, float)
    assert 0 <= acc <= 1


@pytest.mark.parametrize("Model", MODELS)
def test_compute_accuracy_batch_type_bound(Model):
    model = Model(4, 3)
    features = np.ones((256, 4))
    labels = npr.randint(3, size=(256, 3))
    acc = model.compute_accuracy_batch(features, labels, batch_size=32)
    assert isinstance(acc, float)
    assert 0 <= acc <= 1


@pytest.mark.parametrize("Model", MODELS)
def test_compute_loss_batch_type(Model):
    model = Model(4, 3)
    features = np.ones((256, 4))
    labels = npr.randint(3, size=(256, 3))
    loss = model.compute_loss_batch(features, labels, batch_size=32)
    assert isinstance(loss, float)


@pytest.mark.parametrize("Model", MODELS)
def test_compute_gradients_batch_shape(Model):
    model = Model(4, 3)
    features = np.ones((256, 4))
    labels = npr.randint(3, size=(256, 3))
    grads = model.compute_gradients_batch(features, labels, batch_size=32)
    assert grads.shape == (4, 3)


@pytest.mark.parametrize("Model", MODELS)
def test_compute_backprop_batch_type_bound_shape(Model):
    model = Model(4, 3)
    features = np.ones((256, 4))
    labels = npr.randint(3, size=(256, 3))
    loss, grad, acc = model.compute_backprop_batch(features, labels)
    assert isinstance(loss, float)
    assert grad.shape == (4, 3)
    assert isinstance(acc, float)
    assert 0 <= acc <= 1
