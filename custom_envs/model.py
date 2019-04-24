
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


class ModelTF:
    def __init__(self, feature_size, num_of_labels, seed=None):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(seed)
            self.features = tf.placeholder(tf.float32, [None, feature_size])
            self.labels = tf.placeholder(tf.float32, [None, num_of_labels])
            self._weights = tf.Variable(tf.zeros([feature_size, num_of_labels]))

            self.logits = self.features @ self._weights
            self.output = tf.nn.softmax(self.logits)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2
            self.objective = tf.reduce_mean(cross_entropy(labels=self.labels,
                                                          logits=self.logits))
            self.gradients = tf.gradients(self.objective, self._weights)
            decision = tf.argmax(self.output, -1)
            true_label = tf.argmax(self.labels, -1)
            comparisons = tf.cast(tf.equal(decision, true_label), tf.float32)
            self.accuracy = tf.reduce_mean(comparisons)
            
            random_weight = tf.random.normal((feature_size, num_of_labels))
            self.weight_reset_ops = self._weights.assign(random_weight)
            self.weight_ph = tf.placeholder(tf.float32,
                                            [feature_size, num_of_labels])
            self.weight_set_ops = self._weights.assign(self.weight_ph)

            init = tf.global_variables_initializer()
        self.session = tf.Session(graph=self.graph)
        self.session.run(init)

    def reset(self, np_random=npr):
        self.session.run(self.weight_reset_ops)

    @property
    def size(self):
        return np.prod(self.weights.shape)

    @property
    def weights(self):
        return self.session.run(self._weights)

    def forward(self, features):
        fd = {self.features: features}
        return self.session.run(self.output, feed_dict=fd)

    def compute_loss(self, features, labels, acts=None):
        fd = {self.features: features, self.labels: labels}
        return self.session.run(self.objective, feed_dict=fd)

    def compute_gradients(self, features, labels, acts=None):
        fd = {self.features: features, self.labels: labels}
        return self.session.run(self.gradients, feed_dict=fd)

    def compute_accuracy(self, features, labels, acts=None):
        fd = {self.features: features, self.labels: labels}
        return self.session.run(self.accuracy, feed_dict=fd)

    def compute_backprop(self, features, labels):
        fd = {self.features: features, self.labels: labels}
        ops = (self.objective, self.gradients, self.accuracy)
        return self.session.run(ops, feed_dict=fd)

    def set_weights(self, weights):
        fd = {self.weight_ph: weights}
        self.session.run(self.weight_set_ops, feed_dict=fd)

if __name__ == '__main__':
    from time import time
    import numpy.random as npr
    from custom_envs import load_data

    NUM_OF_EPOCHS = 200

    data = load_data()
    features = data[..., :-1]
    labels, num_of_labels = to_onehot(data[..., -1])


    model = Model(features.shape[-1], num_of_labels)

    loss, gradients, acc = model.compute_backprop(features, labels)
    begin = time()
    for i in range(NUM_OF_EPOCHS):
        indices = npr.choice(range(150), size=32, replace=False)
        feat_batch = features[indices, ...]
        lab_batch = labels[indices, ...]
        loss, gradients, acc = model.compute_backprop(feat_batch, lab_batch)
        model.set_weights(model.weights - 1e-3 * gradients)
    time_elapsed = time() - begin
    loss = model.compute_loss(features, labels)
    print('Loss: ', loss)
    print('Time Elapsed: ', time_elapsed)
    print(time_elapsed / NUM_OF_EPOCHS, ' seconds per epoch')

    model = ModelTF(features.shape[-1], num_of_labels)

    loss, gradients, acc = model.compute_backprop(features, labels)
    begin = time()
    for i in range(NUM_OF_EPOCHS):
        indices = npr.choice(range(150), size=32, replace=False)
        feat_batch = features[indices, ...]
        lab_batch = labels[indices, ...]
        loss, gradients, acc = model.compute_backprop(feat_batch, lab_batch)
        model.set_weights(model.weights - 1e-3 * gradients[0])
    time_elapsed = time() - begin
    loss = model.compute_loss(features, labels)
    print('Loss: ', loss)
    print('Time Elapsed: ', time_elapsed)
    print(time_elapsed / NUM_OF_EPOCHS, ' seconds per epoch')