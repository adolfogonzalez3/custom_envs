
from pathlib import Path

import numpy as np

from custom_envs.utils.utils_common import normalize, resize_all

def load_data(name='iris'):
    '''
    Load a data set.

    '''
    path = Path(__file__).resolve().parent
    if name == 'iris':
        data = np.load((path / name).with_suffix('.npz'))['data']
        return data[..., :-1], data[..., -1]
    elif name == 'mnist':
        data = np.load((path / name).with_suffix('.npz'))['data']
        return normalize(resize_all(data[..., :-1])), data[..., -1]
    elif name == 'skin':
        data = np.loadtxt(path / 'skin.txt', delimiter='\t')
        features = np.zeros((data.shape[0], 4))
        features[:, :3] = normalize(data[..., :-1])
        return features, data[..., -1]
    else:
        raise RuntimeError('No such data set named: {}'.format(name))
