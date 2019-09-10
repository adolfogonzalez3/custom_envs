'''A class for loading data in batches in memory.'''
import math

from custom_envs.dataset import BatchType, DataSet
from custom_envs.utils.utils_common import shuffle


class InMemoryDataSet(DataSet):
    '''A class used to get batches from a data set stored in memory.'''

    def __init__(self, features, targets, batch_size=None):
        assert len(features) == len(targets)
        self.features = features
        self.targets = targets
        self.batch_size = len(features) if batch_size is None else batch_size

    def on_epoch_end(self):
        '''Called at the end of an epoch.'''
        self.features, self.targets = shuffle(self.features, self.targets)

    def __len__(self):
        return math.ceil(len(self.features) / self.batch_size)

    def __getitem__(self, idx):
        begin = idx*self.batch_size
        end = begin + self.batch_size
        idx = slice(begin, end if idx < len(self) else None)
        return BatchType(self.features[idx], self.targets[idx])

    @property
    def feature_shape(self):
        '''Return the shape of the features.'''
        return self.features.shape[1:]

    @property
    def target_shape(self):
        '''Return the shape of the targets.'''
        return self.targets.shape[1:]
