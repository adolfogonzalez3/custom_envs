'''A module containing an abstract class for representing data sets.'''

from abc import abstractmethod
from collections import namedtuple

from tensorflow.keras.utils import Sequence

BatchType = namedtuple('BatchType', ['features', 'labels'])


class DataSet(Sequence):
    '''An abstract class for representing data sets.'''

    @property
    @abstractmethod
    def feature_shape(self):
        '''Return the shape of the features.'''

    @property
    @abstractmethod
    def target_shape(self):
        '''Return the shape of the targets.'''
