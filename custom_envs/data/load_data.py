'''A module with utilities to load data sets.'''
from pathlib import Path

import numpy as np

from custom_envs.data.sequence import InMemorySequence
from custom_envs.utils.utils_image import resize_all
from custom_envs.utils.utils_common import to_onehot, normalize


def load_data(name='iris', batch_size=None, num_of_labels=None):
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
        features = data[..., :-1]
        labels, _ = to_onehot(data[..., -1], num_of_labels)
    elif name == 'mnist':
        data = np.load((path / name).with_suffix('.npz'))['data']
        features = normalize(resize_all(data[..., :-1]))
        labels, _ = to_onehot(data[..., -1], num_of_labels)
    elif name == 'skin':
        data = np.loadtxt(path / 'skin.txt', delimiter='\t')
        features = np.zeros((data.shape[0], 4))
        features[:, :3] = normalize(data[..., :-1])
        labels, _ = to_onehot(data[..., -1], num_of_labels)
    else:
        raise RuntimeError('No such data set named: {}'.format(name))
    sequence = InMemorySequence(features, labels, batch_size)

    return sequence
