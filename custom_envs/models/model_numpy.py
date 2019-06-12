
import numexpr
import numpy as np
import numpy.random as npr

from custom_envs.models.model import ModelBase
from custom_envs.utils.utils_math import softmax, cross_entropy

class ModelNumpy(ModelBase):
    '''
    A model that uses a numpy backend.
    '''
    def __init__(self, feature_size, num_of_labels):
        '''
        Create a model.
        
        :param feature_size: (int) The number of features.
        :param num_of_labels: (int) The number of labels.
        '''
        self._weights = npr.normal(size=(feature_size, num_of_labels))

    @property
    def weights(self):
        return self._weights

    @property
    def size(self):
        '''
        Return the number of parameters in the model.
        '''
        return self.weights.size

    def reset(self, np_random=npr):
        '''
        Reset the model's parameters with a normal distribution.
        '''
        self._weights = np_random.normal(size=self.weights.shape)

    def forward(self, features):
        '''
        Forward pass of the model.
        '''
        return softmax(features @ self.weights)

    def compute_loss(self, features, labels, acts=None):
        '''
        Compute the loss given the features and labels.
        '''
        if acts is None:
            acts = self.forward(features)
        objective = cross_entropy(acts, labels)
        return objective

    def compute_gradients(self, features, labels, acts=None):
        '''
        Compute the gradients of all parameters in respect to the cost.
        '''
        if acts is None:
            acts = self.forward(features)
        #gradient = numexpr.evaluate('(-acts*(1-acts)) * (labels - acts)')
        gradient = numexpr.evaluate('(acts - labels)')
        return features.T @ gradient

    def compute_accuracy(self, features, labels, acts=None):
        '''
        Compute the accuracy.
        '''
        if acts is None:
            acts = self.forward(features)
        predictions = np.argmax(acts, axis=1)
        true = np.argmax(labels, axis=1)
        return np.mean(predictions == true)

    def compute_backprop(self, features, labels):
        '''
        Compute loss, gradients, and accuracy.
        '''
        acts = self.forward(features)
        loss = self.compute_loss(features, labels, acts=acts)
        gradients = self.compute_gradients(features, labels, acts=acts)
        accuracy = self.compute_accuracy(features, labels, acts=acts)
        return loss, gradients, accuracy

    def set_weights(self, weights):
        '''
        Set the weights of the model.
        '''
        self._weights = weights

    def get_weights(self):
        '''
        Get the weights of the model.
        '''
        return self.weights
