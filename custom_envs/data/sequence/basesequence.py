'''A module containing an abstract class for loading data sets.'''

from collections import namedtuple
from abc import ABC, abstractmethod

BatchType = namedtuple('BatchType', ['features', 'labels'])


class BaseSequence(ABC):
    '''An abstract class for loading data sets.'''

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass
