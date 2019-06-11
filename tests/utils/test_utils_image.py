'''Tests utils_image module.'''

import numpy as np
import numpy.random as npr

import custom_envs.utils.utils_image as utils_image


def test_resize_array():
    array = npr.randint(255, size=(1024, 1024))
    resized_array = utils_image.resize_array(array, (16, 16))
    assert np.all(resized_array < 256)
    assert np.all(resized_array >= 0)
    assert resized_array.shape == (16, 16)


def test_resize_array_many():
    arrays = npr.randint(255, size=(256, 1024, 1024))
    resized_array = np.array(utils_image.resize_array_many(arrays, (16, 16)))
    assert len(resized_array) == 256
    assert np.all(resized_array < 256)
    assert np.all(resized_array >= 0)
    assert resized_array.shape == (256, 16, 16)
