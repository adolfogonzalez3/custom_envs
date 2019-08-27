'''A module containing an abstract class for representing data sets.'''

from collections import namedtuple
from abc import abstractmethod

from tensorflow.keras.utils import Sequence

BatchType = namedtuple('BatchType', ['features', 'labels'])


class DataSet(Sequence):
    '''An abstract class for representing data sets.'''

    @abstractmethod
    @property
    def feature_shape(self):
        '''Return the shape of the features.'''

    @abstractmethod
    @property
    def target_shape(self):
        '''Return the shape of the targets.'''
