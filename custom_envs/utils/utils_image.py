'''Utilities for manipulating images.'''
import numpy as np
from PIL import Image


def resize_array(array, shape, resample=0):
    '''
    Resize an array using pillow.

    :param array: (numpy.array) The array to resize.
    :param shape: (tuple) The shape of the resulting array.
    :return: (numpy.array) A numpy array.
    '''
    return np.asarray(Image.fromarray(array).resize(shape, resample))


def resize_array_many(arrays, shape, resample=0):
    '''
    Resize many arrays using pillow.

    :param array: (numpy.array) The array to resize.
    :param shape: (tuple) The shape of the resulting array.
    :return: ([numpy.array]) A list of numpy array.
    '''
    return [resize_array(sample, shape, resample) for sample in arrays]
