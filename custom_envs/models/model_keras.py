
import numpy as np
import numpy.random as npr
import tensorflow.keras as keras
import tensorflow.keras.layers as k_layers

from custom_envs.models.model import ModelBase

class ModelKeras(ModelBase):
    '''
    A model that uses a numpy backend.
    '''
    def __init__(self, feature_size, num_of_labels, seed=None):
        target = k_layers.Input((1,))
        model = keras.Sequential()
        model.add(k_layers.Dense(num_of_labels, input_shape=(feature_size,),
                                 use_bias=False))
        loss_function = keras.losses.CategoricalCrossentropy()
        loss = loss_function(target, model.output)
        grads = keras.backend.gradients(loss, model.weights)
        get_grads = keras.backend.function((model.input, target), grads)
        model.compile('sgd', loss_function, metrics=['acc'])
        self.model = model
        self.loss_function = keras.backend.function((model.input, target), loss)
        self.grad_function = get_grads

    def reset(self, np_random=npr):
        '''
        Reset the model's parameters with a normal distribution.
        '''
        self.model.set_weights([np_random.normal(size=w.shape.as_list())
                                for w in self.model.weights])

    @property
    def size(self):
        '''
        Return the number of parameters in the model.
        '''
        return sum([keras.backend.count_params(w) for w in self.model.weights])

    @property
    def weights(self):
        return self.get_weights()

    def forward(self, features):
        '''
        Forward pass of the model.
        '''
        return self.model.predict(features, verbose=0)

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
        return self.grad_function((features, labels))[0]

    def compute_accuracy(self, features, labels, acts=None):
        '''
        Compute the accuracy.
        '''
        labels = np.argmax(labels, axis=-1)
        return float(self.model.test_on_batch(features, labels)[1])

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
        self.model.set_weights([weights])

    def get_weights(self):
        '''
        Get the weights of the model.
        '''
        return self.model.get_weights()[0]

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
        return float(loss), grad, float(accu)
