
import numpy as np
import numpy.random as npr
import numexpr

from custom_envs.utils import softmax, cross_entropy, to_onehot

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.backend import function, gradients

class Model:
    def __init__(self, feature_size, num_of_labels):
        self.weights = npr.normal(size=(feature_size, num_of_labels))

    def reset(self, np_random=npr):
        self.weights = np_random.normal(size=self.weights.shape)

    @property
    def size(self):
        return self.weights.size

    def forward(self, features):
        return softmax(features @ self.weights)

    def compute_loss(self, features, labels, acts=None):
        if acts is None:
            acts = self.forward(features)
        objective = cross_entropy(acts, labels)
        return objective

    def compute_gradients(self, features, labels, acts=None):
        if acts is None:
            acts = self.forward(features)
        #gradient = numexpr.evaluate('(-acts*(1-acts)) * (labels - acts)')
        gradient = numexpr.evaluate('(acts - labels)')
        return features.T @ gradient

    def compute_accuracy(self, features, labels, acts=None):
        if acts is None:
            acts = self.forward(features)
        predictions = np.argmax(acts, axis=1)
        true = np.argmax(labels, axis=1)
        return np.mean(predictions == true)

    def compute_backprop(self, features, labels):
        acts = self.forward(features)
        loss = self.compute_loss(features, labels, acts=acts)
        gradients = self.compute_gradients(features, labels, acts=acts)
        accuracy = self.compute_accuracy(features, labels, acts=acts)
        return loss, gradients, accuracy


    def set_weights(self, weights):
        self.weights = weights


class KerasModel:
    def __init__(self, feature_size, num_of_labels):
        self.model = Sequential()
        self.model.add(Dense(num_of_labels, input_shape=(feature_size,),
                             activation='softmax'))
        self.shapes = [w.shape for w in self.model.get_weights()]
        self.size = np.sum([np.prod(shape) for shape in self.shapes])
        self.model.compile('SGD', tf.losses.softmax_cross_entropy)
        #self._loss = function(self.model.input, self.model.losses)


    def reset(self, np_random=npr):
        weights = self.model.get_weights()
        self.model.set_weights(npr.random(weights.shape))

    def size(self):
        return self.size

    def forward(self, features):
        return self.model.predict_on_batch(features)

    def compute_loss(self, features, labels, acts=None):
        if acts is None:    
            acts = self.forward(features)
        return self._loss(features)

    def compute_gradients(self, features, labels, acts=None):
        if acts is None:
            acts = self.forward(features)
        #gradient = numexpr.evaluate('(-acts*(1-acts)) * (labels - acts)')
        gradient = numexpr.evaluate('(acts - labels)')
        return features.T @ gradient

    def compute_accuracy(self, features, labels, acts=None):
        if acts is None:
            acts = self.forward(features)
        predictions = np.argmax(acts, axis=1)
        true = np.argmax(labels, axis=1)
        return np.mean(predictions == true)

    def compute_backprop(self, features, labels):
        acts = self.forward(features)
        loss = self.compute_loss(features, labels, acts=acts)
        gradients = self.compute_gradients(features, labels, acts=acts)
        accuracy = self.compute_accuracy(features, labels, acts=acts)
        return loss, gradients, accuracy


    def set_weights(self, weights):
        self.weights = weights

if __name__ == '__main__':
    import numpy.random as npr
    from custom_envs import load_data

    data = load_data()
    features, labels = np.hsplit(data, [-1])
    labels, num_of_labels = to_onehot(labels)
    model = KerasModel(features.shape[1], num_of_labels)
    #print(model.forward(features))
    #function(model.model.input, model.model.losses)
    print(model.model.input)
    print(model.model.losses)


def what():
    loss, gradients, acc = model.compute_backprop(features, labels)
    print(loss, acc)
    print(gradients)
    for i in range(40):
        indices = npr.choice(range(150), size=32, replace=False)
        feat_batch = features[indices, ...]
        lab_batch = labels[indices, ...]
        loss, gradients, acc = model.compute_backprop(feat_batch, lab_batch)
        print('Loss: ', loss, 'Acc: ', acc)
        model.set_weights(model.weights - 1e-3 * gradients)
    loss = model.compute_loss(features, labels)
    print('Loss: ', loss)