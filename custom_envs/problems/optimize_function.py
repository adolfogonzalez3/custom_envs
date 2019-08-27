'''A module containing a Model subclass implemented with keras.'''
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import custom_envs.utils.utils_common as common
import custom_envs.utils.utils_tf as utils_tf
from custom_envs.utils.utils_functions import compute_rosenbrock
from custom_envs.problems import BaseProblem

NetTensors = namedtuple('NetTensors', ['gradient', 'loss', 'weight'])
NetParamObjs = namedtuple('NetParamObjs', ['inputs', 'setters'])


class OptimizeFunction(BaseProblem):
    '''A class for implementing problems which use tensorflow.'''

    def __init__(self, function=None, initial_points=None, ndims=2):
        '''
        Create a problem with the goal of optimizing a NN for a data set.

        :param function: () A function which the output of a tensorflow graph.
        :param initial_points: (sequence) A sequence of numbers that represents
            the starting point for the variables. If None then the variables
            are chosen randomly.
        :param ndims: (int) Represents the number of dimensions of the
            function.
        '''
        if initial_points:
            self.initial_points = initial_points
        else:
            self.initial_points = [tf.random_normal_initializer()]*ndims
        if not function:
            function = compute_rosenbrock
            self.initial_points = [-1.9, 2.]
        assert len(self.initial_points) == ndims
        with tf.name_scope('parameters'):
            self.variables = tuple(
                tf.get_variable(
                    'param_{:d}'.format(i),
                    initializer=tf.constant(initial_point)
                )
                for i, initial_point in enumerate(self.initial_points)
            )
        with tf.name_scope('output'):
            loss = tf.reduce_mean(function(*self.variables))
            self.tensors_out = NetTensors(
                tf.gradients(loss, self.variables), loss, self.variables
            )

        param_in = tuple(tf.placeholder(tf.float32) for i in range(ndims))
        param_update = tuple(
            param.assign(pin) for param, pin in zip(self.variables, param_in)
        )
        self._size = ndims
        self.param_info = NetParamObjs(param_in, param_update)
        self.reset_init = tf.global_variables_initializer()
        self.reset()

    @classmethod
    def create(cls, function=None, initial_points=None, ndims=2):
        '''
        Create a problem with the goal of optimizing a NN for a data set.

        :param function: () A function which the output of a tensorflow graph.
        :param initial_points: (sequence) A sequence of numbers that represents
            the starting point for the variables. If None then the variables
            are chosen randomly.
        :param ndims: (int) Represents the number of dimensions of the
            function.
        :return: (OptimizeFunction)
        '''
        return utils_tf.wrap_in_session(cls)(function, initial_points, ndims)

    @property
    def size(self):
        '''Return the number of parameters.'''
        return self._size

    def run(self, tensors, feed=None):
        '''
        Run graph operations in the session.

        :param tensors: (Tensorflow tensor) Tensors to retrieve.
        :param feed: (dict) A dictionary of parameters to input.
        :return: () Return retrieved data from tensors.
        '''
        return tf.get_default_session().run(tensors, feed_dict=feed)

    def next(self):
        '''
        Get the next batch of data from the data set.
        '''

    def reset(self):
        '''
        Reset the Problem to its initial point.
        '''
        self.run(self.reset_init)

    def get_gradient(self):
        '''
        Retrieve the current gradient for the problem.
        '''
        return self.run(self.tensors_out.gradient)

    def get_loss(self):
        '''
        Retrieve the current loss for the problem.
        '''
        return self.run(self.tensors_out.loss)

    def get_parameters(self):
        '''
        Retrieve the current parameters for the problem.
        '''
        return self.run(self.tensors_out.weight)

    def set_parameters(self, parameters):
        '''
        Set the new parameters for the problem.
        '''
        feed_dict = {
            pl: param for pl, param in zip(self.param_info.inputs, parameters)
        }
        self.run(self.param_info.setters, feed_dict)

    def get(self):
        '''
        Retrieve the gradient, loss, and parameters for the problem.
        '''
        gradient, loss, weight = self.run(self.tensors_out)
        #gradient = common.flatten_arrays(gradient)
        #weight = common.flatten_arrays(weight)
        return gradient, loss, weight
