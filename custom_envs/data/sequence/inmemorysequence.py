'''A class for loading data in batches in memory.'''

import math

from custom_envs.utils.utils_common import shuffle
from custom_envs.data.sequence import BaseSequence, BatchType


class InMemorySequence(BaseSequence):
    '''A class used to get batches from a data set stored in memory.'''

    def __init__(self, features, labels, batch_size=None):
        assert len(features) == len(labels)
        self.features = features
        self.labels = labels
        self.batch_size = len(features) if batch_size is None else batch_size
        self._feature_shape = self.features.shape[1:]
        self._label_shape = self.labels.shape[1:]

    def shuffle(self):
        self.features, self.labels = shuffle(self.features, self.labels)

    def __len__(self):
        return math.ceil(len(self.features) / self.batch_size)

    def __getitem__(self, idx):
        begin = idx*self.batch_size
        end = begin + self.batch_size
        idx = slice(begin, end if idx < len(self) else None)
        return BatchType(self.features[idx], self.labels[idx])

    @property
    def feature_shape(self):
        return self._feature_shape

    @property
    def label_shape(self):
        return self._label_shape
