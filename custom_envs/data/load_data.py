'''A module with utilities to load data sets.'''
import lzma
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn import datasets

from custom_envs.dataset import InMemoryDataSet
from custom_envs.utils.utils_image import resize_array_many, convert_many
from custom_envs.utils.utils_common import to_onehot
from custom_envs.utils.utils_math import normalize


def load_mnist(name='fashion', kind='train'):
    """Load MNIST data from `path`"""
    path = Path(__file__).resolve().parent
    labels_path = path / name / ('%s-labels-idx1-ubyte.xz' % kind)
    images_path = path / name / ('%s-images-idx3-ubyte.xz' % kind)

    with lzma.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with lzma.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def load_emnist(name='emnist', kind='train'):
    """Load EMNIST data from `path`"""
    path = Path(__file__).resolve().parent / name / 'digits'
    labels_path = path / ('emnist-digits-%s-labels-idx1-ubyte.xz' % kind)
    images_path = path / ('emnist-digits-%s-images-idx3-ubyte.xz' % kind)

    with lzma.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with lzma.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def load_data(name='iris', batch_size=32, num_of_labels=None):
    '''
    Load a data set.

    :param name: (str) Name of the data set to load.
    :param batch_size: (int or None) Size of the mini batches if int otherwise
                                     if None then loads entire data set.
    :param num_of_labels: (int or None) Number of integers to represent labels
                                        in one hot notation if integer
                                        otherwise if None then infer from
                                        data.
    :return: (subclass of BaseSequence)
    '''
    path = Path(__file__).resolve().parent
    if name == 'iris':
        data = np.load((path / name).with_suffix('.npz'))['data']
        features = normalize(data[..., :-1])
        labels, _ = to_onehot(data[..., -1], num_of_labels)
    elif name == 'mnist':
        features, labels = load_mnist('mnist')
        features = [f.reshape((28, 28)) for f in features]
        features = np.reshape(resize_array_many(features, (7, 7)),
                              (len(labels), -1))
        features = normalize(features)
        labels, _ = to_onehot(labels, num_of_labels)
    elif name == 'mnist-test':
        features, labels = load_mnist('mnist', 't10k')
        features = [f.reshape((28, 28)) for f in features]
        features = np.reshape(resize_array_many(features, (7, 7)),
                              (len(labels), -1))
        features = normalize(features)
        labels, _ = to_onehot(labels, num_of_labels)
    elif name == 'skin':
        data = np.loadtxt(path / 'skin.txt', delimiter='\t')
        features = np.zeros((data.shape[0], 4))
        features[:, :3] = normalize(data[..., :-1])
        labels, _ = to_onehot(data[..., -1], 3)
    elif name == 'fashion':
        features, labels = load_mnist()
        features = [f.reshape((28, 28)) for f in features]
        features = np.reshape(resize_array_many(features, (7, 7)),
                              (len(labels), -1))
        features = normalize(features)
        labels, _ = to_onehot(labels, num_of_labels)
    elif name == 'emnist-digits':
        features, labels = load_emnist()
        features = [f.reshape((28, 28)) for f in features]
        features = np.reshape(resize_array_many(features, (7, 7)),
                              (len(labels), -1))
        features = normalize(features)
        labels, _ = to_onehot(labels, num_of_labels)
    elif name == 'cifar-10':
        (images_tr, labels_tr), _ = tf.keras.datasets.cifar10.load_data()
        images_tr = convert_many(images_tr, mode='L')
        images_tr = resize_array_many(images_tr, (7, 7))
        features = np.reshape(images_tr, (len(labels_tr), -1))
        features = normalize(features)
        labels, _ = to_onehot(labels_tr)
    elif name == 'random_gaussians':
        features, labels = datasets.make_classification()
        labels, _ = to_onehot(labels, 2)
    else:
        raise RuntimeError('No such data set named: {}'.format(name))
    sequence = InMemoryDataSet(features, labels, batch_size)

    return sequence
