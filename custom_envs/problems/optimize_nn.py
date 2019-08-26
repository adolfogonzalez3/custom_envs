'''A module containing a Model subclass implemented with keras.'''
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import custom_envs.utils.utils_common as common
import custom_envs.utils.utils_tf as utils_tf
from custom_envs.problems import BaseProblem
from custom_envs.data import load_data

NetInput = namedtuple('NetInput', ['feature', 'target'])
NetTensors = namedtuple('NetTensors', ['gradient', 'loss', 'weight'])
NetBatch = namedtuple('NetBatch', ['feature', 'target'])
NetParamObjs = namedtuple('NetParamObjs', ['inputs', 'setters', 'shapes'])


class OptimizeNN(BaseProblem):
    '''A class for implementing problems which use tensorflow.'''

    def __init__(self, model_fn=None, data_set=None):
        '''
        Create a problem with the goal of optimizing a NN for a data set.

        :param model_fn: () A function which returns a keras model.
        :param data_set: () A function which returns a data set.
        '''
        self.data_set_iter = None
        self.current_batch = None
        if data_set:
            self.data_set = data_set
        else:
            self.data_set = load_data()
        inputs = keras.Input(self.data_set.feature_shape, name='feature')
        target = keras.Input(self.data_set.label_shape, name='target')
        with tf.name_scope('network'):
            if callable(model_fn):
                outputs = model_fn(inputs)
            else:
                outputs = utils_tf.create_neural_net(inputs)
            outputs = keras.layers.Dense(
                self.data_set.label_shape[0], activation='softmax'
            )(outputs)
        weight_tensors = tf.trainable_variables(scope='network')
        with tf.name_scope('output'):
            loss = keras.losses.categorical_crossentropy(target, outputs)
            self.tensors_out = NetTensors(
                tf.gradients(loss, weight_tensors),
                tf.reduce_mean(loss),
                weight_tensors
            )
        shapes = tuple(tuple(w.shape.as_list()) for w in weight_tensors)
        weight_in = tuple(
            tf.placeholder(tf.float32, shape) for shape in shapes
        )
        weight_update = tuple(
            w.assign(p) for w, p in zip(weight_tensors, weight_in)
        )
        self._size = sum([np.prod(shape) for shape in shapes])
        self.param_info = NetParamObjs(weight_in, weight_update, shapes)
        self.net_inputs = NetInput(inputs, target)
        self.reset_init = tf.global_variables_initializer()
        self.reset()

    @classmethod
    def create(cls, *args, **kwargs):
        '''Create an object of the class.'''
        return utils_tf.wrap_in_session(cls)(*args, **kwargs)

    @property
    def size(self):
        '''Return the number of parameters.'''
        return self._size

    def run(self, tensors, feed=True):
        '''
        Run graph operations in the session.

        :param tensors: (Tensorflow tensor) Tensors to retrieve.
        :param feed: (bool) If True feed current batch else nothing.
        :return: () Return retrieved data from tensors.
        '''
        if feed:
            if isinstance(feed, dict):
                feed_dict = feed
            else:
                feed_dict = {
                    tensor: data for tensor, data in
                    zip(self.net_inputs, self.current_batch)
                }
        else:
            feed_dict = None
        return tf.get_default_session().run(tensors, feed_dict=feed_dict)

    def next(self):
        '''
        Get the next batch of data from the data set.
        '''
        try:
            features, targets = next(self.data_set_iter)
        except StopIteration:
            self.data_set.on_epoch_end()
            self.data_set_iter = iter(self.data_set)
            features, targets = next(self.data_set_iter)
        self.current_batch = NetBatch(features, targets)

    def reset(self):
        '''
        Reset the Problem to its initial point.
        '''
        self.run(self.reset_init, feed=False)
        self.data_set_iter = iter(())
        self.next()

    def get_gradient(self):
        '''
        Retrieve the current gradient for the problem.
        '''
        return common.flatten_arrays(self.run(self.tensors_out.gradient))

    def get_loss(self):
        '''
        Retrieve the current loss for the problem.
        '''
        return self.run(self.tensors_out.loss)

    def get_parameters(self):
        '''
        Retrieve the current parameters for the problem.
        '''
        return common.flatten_arrays(
            self.run(self.tensors_out.weight, feed=False)
        )

    def set_parameters(self, parameters):
        '''
        Set the new parameters for the problem.
        '''
        parameters = common.from_flat(parameters, self.param_info.shapes)
        feed_dict = {
            pl: param for pl, param in zip(self.param_info.inputs, parameters)
        }
        self.run(self.param_info.setters, feed_dict)

    def get(self):
        '''
        Retrieve the gradient, loss, and parameters for the problem.
        '''
        gradient, loss, weight = self.run(self.tensors_out)
        gradient = common.flatten_arrays(gradient)
        weight = common.flatten_arrays(weight)
        return gradient, loss, weight
