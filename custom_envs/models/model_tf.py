
import numpy as np
import numpy.random as npr
import tensorflow as tf

class ModelTF:
    '''
    A model that uses a numpy backend.
    '''
    def __init__(self, feature_size, num_of_labels, seed=None):
        '''
        Create a model.
        
        :param feature_size: (int) The number of features.
        :param num_of_labels: (int) The number of labels.
        '''
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
        '''
        Reset the model's parameters with a normal distribution.
        '''
        self.session.run(self.weight_reset_ops)

    @property
    def size(self):
        '''
        Return the number of parameters in the model.
        '''    
        return np.prod(self.weights.shape)

    @property
    def weights(self):
        return self.get_weights()

    def forward(self, features):
        '''
        Forward pass of the model.
        '''
        fd = {self.features: features}
        return self.session.run(self.output, feed_dict=fd)

    def compute_loss(self, features, labels, acts=None):
        '''
        Compute the loss given the features and labels.
        '''
        fd = {self.features: features, self.labels: labels}
        return float(self.session.run(self.objective, feed_dict=fd))

    def compute_gradients(self, features, labels, acts=None):
        '''
        Compute the gradients of all parameters in respect to the cost.
        '''
        fd = {self.features: features, self.labels: labels}
        return self.session.run(self.gradients, feed_dict=fd)[0]

    def compute_accuracy(self, features, labels, acts=None):
        '''
        Compute the accuracy.
        '''
        fd = {self.features: features, self.labels: labels}
        return float(self.session.run(self.accuracy, feed_dict=fd))

    def compute_backprop(self, features, labels):
        '''
        Compute loss, gradients, and accuracy.
        '''
        fd = {self.features: features, self.labels: labels}
        ops = (self.objective, self.gradients, self.accuracy)
        loss, grad, acc = self.session.run(ops, feed_dict=fd)
        return float(loss), grad[0], float(acc)

    def set_weights(self, weights):
        '''
        Set the weights of the model.
        '''
        fd = {self.weight_ph: weights}
        self.session.run(self.weight_set_ops, feed_dict=fd)

    def get_weights(self):
        '''
        Get the weights of the model.
        '''
        return self.session.run(self._weights)
