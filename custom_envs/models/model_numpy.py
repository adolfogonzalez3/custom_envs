'''A module containing a class for model using numpy and numexpr.'''
import numexpr
import numpy as np
import numpy.random as npr

import custom_envs.utils.utils_common as common
from custom_envs.models.model import ModelBase
from custom_envs.utils.utils_math import softmax, cross_entropy


class ModelNumpy(ModelBase):
    '''
    A model that uses a numpy backend.
    '''

    def __init__(self, feature_size, num_of_labels, use_bias=False):
        '''
        Create a model.

        :param feature_size: (int) The number of features.
        :param num_of_labels: (int) The number of labels.
        '''
        feature_size = feature_size + 1 if use_bias else feature_size
        self._weights = npr.normal(size=(feature_size, num_of_labels))
        self.use_bias = use_bias
        self.shapes = [(feature_size, num_of_labels)]
        self._size = self.weights.size

    @property
    def weights(self):
        return common.flatten_arrays([self._weights])

    @property
    def size(self):
        '''
        Return the number of parameters in the model.
        '''
        return self._size

    def reset(self):
        '''
        Reset the model's parameters with a normal distribution.
        '''
        self._weights = npr.normal(size=self._weights.shape)

    def forward(self, features):
        '''
        Forward pass of the model.
        '''
        if self.use_bias:
            features = np.append(features, np.ones((len(features), 1)), axis=1)
        return softmax(features @ self._weights)

    def _compute_loss(self, labels, acts):
        '''
        Compute the loss given the features and labels.
        '''
        return cross_entropy(acts, labels)

    def _compute_gradients(self, features, labels, acts):
        '''
        Compute the gradients of all parameters in respect to the cost.
        '''
        if acts is None:
            acts = self.forward(features)
        if self.use_bias:
            features = np.append(features, np.ones((len(features), 1)), axis=1)
        numvars = {'acts': acts, 'labels': labels}
        gradient = numexpr.evaluate('(acts - labels)', local_dict=numvars)
        return common.flatten_arrays([features.T @ gradient])

    def _compute_accuracy(self, labels, acts):
        '''
        Compute the accuracy.
        '''
        predictions = np.argmax(acts, axis=-1)
        true_labels = np.argmax(labels, axis=-1)
        return np.mean(predictions == true_labels)

    def compute_loss(self, features, labels):
        '''
        Compute the loss given the features and labels.
        '''
        return self._compute_loss(labels, self.forward(features))

    def compute_gradients(self, features, labels):
        '''
        Compute the gradients of all parameters in respect to the cost.
        '''
        acts = self.forward(features)
        return self._compute_gradients(features, labels, acts)

    def compute_accuracy(self, features, labels):
        '''
        Compute the accuracy.

        :param features: (numpy.array) A numpy array of features.
        :param labels: (numpy.array) A numpy array of labels.
        :return: (float) A floating point number.
        '''
        return self._compute_accuracy(labels, self.forward(features))

    def compute_backprop(self, features, labels):
        '''
        Compute loss, gradients, and accuracy.
        '''
        acts = self.forward(features)
        gradients = self._compute_gradients(features, labels, acts=acts)
        accuracy = self._compute_accuracy(labels, acts=acts)
        loss = self._compute_loss(labels, acts=acts)
        return loss, gradients, accuracy

    def set_weights(self, weights):
        '''
        Set the weights of the model.
        '''
        self._weights = common.from_flat(weights, self.shapes)[0]

    def get_weights(self):
        '''
        Get the weights of the model.
        '''
        return common.flatten_arrays([self.weights])
