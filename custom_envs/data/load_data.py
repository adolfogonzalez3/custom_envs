
from pathlib import Path

import numpy as np


def load_data(data_name='iris'):
    name = None
    path = Path(__file__).resolve().parent
    if data_name == 'iris':
        return np.load(path / 'iris.npz')['data']
    elif data_name == 'mnist':
        return np.load(path / 'mnist.npz')['data']
    else:
        raise RuntimeError('No such data set named: {}'.format(data_name))
