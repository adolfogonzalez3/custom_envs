'''
A module that contains the abstract class for implementing models classes.
'''
from abc import ABC, abstractmethod

import numpy.random as npr

from custom_envs.utils.utils_common import batchify_zip

class ModelBase(ABC):
    '''
    An abstract class that is used to implement a model used in the env.
    '''

    @property
    @abstractmethod
    def size(self):
        '''
        Return the number of parameters in the model.
        '''
        pass

    @property
    @abstractmethod
    def weights(self):
        '''
        Return the parameters of the model.
        '''
        pass

    @abstractmethod
    def reset(self, np_random=npr):
        '''
        Reset the model's parameters with a normal distribution.
        '''
        pass

    @abstractmethod
    def forward(self, features):
        '''
        Forward pass of the model.
        '''
        pass

    @abstractmethod
    def compute_loss(self, features, labels):
        '''
        Compute the loss given the features and labels.
        '''
        pass

    @abstractmethod
    def compute_gradients(self, features, labels):
        '''
        Compute the gradients of all parameters in respect to the cost.
        '''
        pass

    @abstractmethod
    def compute_accuracy(self, features, labels, acts=None):
        '''
        Compute the accuracy.
        '''
        pass

    def compute_backprop(self, features, labels):
        '''
        Compute loss, gradients, and accuracy.
        '''
        pass

    @abstractmethod
    def set_weights(self, weights):
        '''
        Set the weights of the model.
        '''
        pass

    @abstractmethod
    def get_weights(self):
        '''
        Get the weights of the model.
        '''
        pass

    def compute_accuracy_batch(self, features, labels, batch_size=32):
        '''Compute the accuracy using batches.'''
        total_correct = 0
        for batch in batchify_zip(features, labels, size=batch_size):
            feature_batch, label_batch = batch
            accuracy = self.compute_accuracy(feature_batch, label_batch)
            total_correct += accuracy*len(feature_batch)
        return total_correct / len(features)

    def compute_loss_batch(self, features, labels, batch_size=32):
        '''Compute the loss using batches.'''
        total_loss = 0
        for batch in batchify_zip(features, labels, size=batch_size):
            feature_batch, label_batch = batch
            loss = self.compute_loss(feature_batch, label_batch)
            total_loss += loss*len(feature_batch)
        return total_loss / len(features)

    def compute_gradients_batch(self, features, labels, batch_size=32):
        '''Compute the gradients using batches.'''
        total_gradients = 0
        for batch in batchify_zip(features, labels, size=batch_size):
            feature_batch, label_batch = batch
            grads = self.compute_gradients(feature_batch, label_batch)
            if total_gradients is None:
                total_gradients = grads
            else:
                total_gradients += grads
        return total_gradients

    def compute_backprop_batch(self, features, labels, batch_size=32):
        '''Compute the loss, gradients, and accuracy using batches.'''
        loss = self.compute_loss_batch(features, labels, batch_size)
        grad = self.compute_gradients_batch(features, labels, batch_size)
        accu = self.compute_accuracy_batch(features, labels, batch_size)
        return loss, grad, accu
