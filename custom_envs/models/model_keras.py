'''A module containing a Model subclass implemented with keras.'''
import numpy as np
import numpy.random as npr
import tensorflow as tf

import custom_envs.utils.utils_common as common
from custom_envs.models.model import ModelBase


class ModelKeras(ModelBase):
    '''
    A model that uses a keras backend.
    '''

    def __init__(self, feature_size, num_of_labels, use_bias=False):
        layers = (feature_size, num_of_labels)
        target = tf.keras.layers.Input((1,))
        model_in = tf.keras.layers.Input([layers[0]])
        tensor = model_in
        layers = [layers[0], 20, layers[-1]]
        layer_activations = []
        for layer in layers[1:-1]:
            layer = tf.keras.layers.Dense(layer, use_bias=use_bias,
                                           activation='relu')
            tensor = layer(tensor)
            layer_activations.extend([1]*sum(np.prod(w.shape) for w in layer.weights))
        layer = tf.keras.layers.Dense(layers[-1], use_bias=use_bias,
                                       activation='softmax')
        tensor = layer(tensor)
        layer_activations.extend([0]*sum(np.prod(w.shape) for w in layer.weights))
        model = tf.keras.Model(inputs=model_in, outputs=tensor)
        loss_function = tf.keras.losses.CategoricalCrossentropy()
        loss = loss_function(target, tensor)
        grads = tf.gradients(loss, model.weights)
        get_grads = tf.keras.backend.function((model.input, target), grads)
        get_loss = tf.keras.backend.function((model.input, target), loss)
        model.compile('sgd', 'sparse_categorical_crossentropy', metrics=['acc'])
        self._size = len(common.flatten_arrays(model.get_weights()))
        self.shapes = tuple(w.shape for w in model.get_weights())
        self.model = model
        self.grad_function = get_grads
        self.loss_function = get_loss
        self.layer_activations = layer_activations

    def reset(self):
        '''
        Reset the model's parameters with a normal distribution.
        '''
        self.model.set_weights([npr.normal(size=shp) for shp in self.shapes])

    @property
    def size(self):
        '''
        Return the number of parameters in the model.
        '''
        return self._size

    @property
    def weights(self):
        return self.get_weights()

    def forward(self, features):
        '''
        Forward pass of the model.
        '''
        return self.model.predict_on_batch(features)

    def compute_loss(self, features, labels):
        '''
        Compute the loss given the features and labels.
        '''
        labels = np.argmax(labels, axis=-1)
        return float(self.model.test_on_batch(features, labels)[0])

    def compute_gradients(self, features, labels):
        '''
        Compute the gradients of all parameters in respect to the cost.
        '''
        labels = np.argmax(labels, axis=-1)
        #print(self.grad_function((features, labels)))
        return common.flatten_arrays(self.grad_function((features, labels)))

    def compute_accuracy(self, features, labels):
        '''
        Compute the accuracy.
        '''
        predictions = np.argmax(self.model.predict_on_batch(features), axis=1)
        labels = np.argmax(labels, axis=-1)
        # print(self.model.predict_on_batch(features))
        # print(predictions)
        # print(labels)
        #print(np.mean(predictions == labels))
        #print(predictions.shape, labels.shape)
        #return float(self.model.test_on_batch(features, labels)[1])
        return np.mean(predictions == labels)

    def compute_backprop(self, features, labels):
        '''
        Compute loss, gradients, and accuracy.
        '''
        loss = self.compute_loss(features, labels)
        grad = self.compute_gradients(features, labels)
        acc = self.compute_accuracy(features, labels)
        return loss, grad, acc

    def set_weights(self, weights):
        '''
        Set the weights of the model.
        '''
        self.model.set_weights(common.from_flat(weights, self.shapes))

    def get_weights(self):
        '''
        Get the weights of the model.
        '''
        return common.flatten_arrays(self.model.get_weights())

    def compute_accuracy_batch(self, features, labels, batch_size=32):
        '''Compute the accuracy using batches.'''
        labels = np.argmax(labels, axis=-1)
        return float(self.model.evaluate(features, labels, verbose=0,
                                         batch_size=batch_size)[1])

    def compute_loss_batch(self, features, labels, batch_size=32):
        '''Compute the loss using batches.'''
        labels = np.argmax(labels, axis=-1)
        return float(self.model.evaluate(features, labels, verbose=0,
                                         batch_size=batch_size)[0])

    def compute_backprop_batch(self, features, labels, batch_size=32):
        '''Compute the loss, gradients, and accuracy using batches.'''
        labs = np.argmax(labels, axis=-1)
        loss, accu = self.model.evaluate(features, labs, verbose=0,
                                         batch_size=batch_size)
        grad = self.compute_gradients_batch(features, labels, batch_size)
        return float(loss), common.flatten_arrays(grad), float(accu)

    #def get_types(self):
    #    return [np.full(shp, )]
