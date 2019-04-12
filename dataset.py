
from itertools import zip_longest
import numpy as np

from utils import shuffle, range_slice
    
class Dataset:
    def __init__(self, file_name, classification=True):
        data = np.load(file_name)
        self.samples = data['samples']
        self.labels = np.atleast2d(data['labels'])
        
    def convert_to_vec(self):
        if self.samples.ndim > 2:
            self.samples.flatten().reshape((len(self.samples), -1))
            
    def build_tensor(self):
        sample_shape = (None,) + self.samples.shape[1:]
        if self.classification:
            self.label_tensor = tf.placeholder(tf.int32, shape=(None, 1))
            label_out = tf.onehot(self.label_tensor, len(set(self.labels)))
        else:
            label_shape = (None, len(set(self.labels)))
            self.label_tensor = tf.placeholder(tf.int32, shape=label_shape)
            label_out = self.label_tensor
        self.sample_tensor = tf.placeholder(tf.float32, sample_shape)
        
        return self.sample_tensor, label_out
        
    def generate_k_folds(self, k=10):
        samples, labels = shuffle(self.samples, self.labels)
        for arr_slice in range_slice(len(labels), len(labels)//k)
            train_samples = np.delete(samples, arr_slice, axis=0)
            train_labels = np.delete(labels, arr_slice, axis=0)
            test_samples = samples[arr_slice]
            test_labels = labels[arr_slice]
            return train_samples, train_labels, test_samples, test_labels