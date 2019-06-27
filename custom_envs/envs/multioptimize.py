"""The MultiOptimize environment."""

import numpy as np
from gym.spaces import Box, Dict

from custom_envs import load_data
#from custom_envs.models import ModelNumpy as Model
from custom_envs.models import ModelKeras as Model
from custom_envs.envs import BaseMultiEnvironment
from custom_envs.utils.utils_common import History, enzip

# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py


class MultiOptimize(BaseMultiEnvironment):

    """
    Summary:
    The optimize environment requires an agent to reduce the
    objective function of a target neural network on some dataset by changing
    the per parameter learning rate. The action of the agent is used as the
    the per parameter learning rate for a vanilla gradient descent algorithm.

    Target Network:
    The target network is a simple multiclass softmax classifier. The
    neural network operations are written with numpy.

    action_space: The number of dimensions is equal to the number of
                  parameters in the target network.
    observation_space: The observation consists of the current parameters of
                       the target network, the current objective, and the
                       current gradient of the objective with respect to
                       the target network's parameters.
    """

    def __init__(self, data_set='iris', batch_size=None, version=1,
                 max_batches=400, max_history=5):
        super().__init__()
        self.sequence = load_data(data_set, batch_size)
        num_of_labels = self.sequence.label_shape[0]
        feature_size = self.sequence.feature_shape[0]
        self.model = Model(feature_size, num_of_labels, use_bias=True)
        # self.model = Model((feature_size, num_of_labels), use_bias=True)
        model_shape = self.model.weights.shape
        self.history = History(max_history, losses=(), gradients=model_shape,
                               weights=model_shape)
        if version == 0:
            self.adjusted_history = History(1, gradients=model_shape)
            state_size = 1
        elif version == 1:
            self.adjusted_history = History(max_history, losses=(),
                                            gradients=model_shape)
            state_size = 2*max_history
        elif version == 2:
            self.adjusted_history = History(1, weights=model_shape,
                                            losses=(), gradients=model_shape)
            state_size = 3
        elif version == 3:
            self.adjusted_history = History(max_history, weights=model_shape,
                                            losses=(), gradients=model_shape)
            state_size = 3*max_history
        elif version == 4:
            self.adjusted_history = History(max_history, gradients=model_shape)
            state_size = max_history + 1
        self.observation_space = Dict({MultiOptimize.AGENT_FMT.format(i):
                                       Box(low=-1e3, high=1e3,
                                           dtype=np.float32,
                                           shape=(state_size,))
                                       for i in range(self.model.size)})
        self.action_space = Dict({MultiOptimize.AGENT_FMT.format(i):
                                  Box(low=-1e3, high=1e3,
                                      dtype=np.float32,
                                      shape=(1,))
                                  for i in range(self.model.size)})
        self.seed()
        self.max_history = max_history
        self.max_batches = max_batches
        self.version = version

    def base_reset(self):
        self.adjusted_history.reset()
        self.model.reset()
        if True:
            seq_idx = len(self.sequence) - 1
            features, labels = self.sequence[seq_idx]
            loss, grad, _ = self.model.compute_backprop(features, labels)
            #grad = grad / len(features)
            self.history.reset(losses=loss, gradients=grad,
                               weights=self.model.weights)
        else:
            self.history.reset()
        states = self.adjusted_history.build_multistate()
        #states = {MultiOptimize.AGENT_FMT.format(i): list(values)
        #          for i, values in enumerate(states)}
        states = {MultiOptimize.AGENT_FMT.format(i): list(v) + [a]
                  for i, v, a in enzip(states, self.model.layer_activations)}
        return states

    def base_step(self, action):
        action = np.reshape([action[MultiOptimize.AGENT_FMT.format(i)]
                             for i in range(self.model.size)], (-1,))
        seq_idx = self.current_step % len(self.sequence)
        if seq_idx == 0:
            self.sequence.shuffle()
        features, labels = self.sequence[seq_idx]
        new_weights = self.model.weights - action
        self.model.set_weights(new_weights)
        loss, grad, accu = self.model.compute_backprop(features, labels)
        #grad = grad / len(features)
        self.history.append(losses=loss, gradients=grad, weights=new_weights)
        past_losses = self.history['losses']
        past_grads = self.history['gradients']
        past_weights = self.history['weights']

        if self.current_step < 3:
            adjusted_loss = 1
            adjusted_wght = 1
            adjusted_grad = np.ones_like(grad)
        else:
            prev_loss = np.abs(past_losses[1] - past_losses[2])
            adjusted_loss = np.divide(past_losses[0] - past_losses[1],
                                      prev_loss + 1e-1)
            prev_wght = np.abs(past_weights[1] - past_weights[2])
            abs_wght = np.abs(past_weights[0] - past_weights[1])
            adjusted_wght = np.divide(prev_wght, abs_wght + 1e-8)
            prev_grad = np.abs(past_grads[1] - past_grads[2])
            adjusted_grad = np.divide(past_grads[0] - past_grads[1],
                                      prev_grad + 1e-1)
        #adjusted_loss = np.divide(past_losses[0] - past_losses[1],
        #                          np.abs(past_losses[1]) + 1e-1)
        #adjusted_loss = np.divide(past_losses[0],
        #                          np.abs(past_losses[1]) + 1e-1)
        #adjusted_loss += np.mean(np.abs(new_weights))*1e-2
        adjusted_grad = np.divide(past_grads[0], np.abs(past_grads[1]) + 1e-1)
        
        if self.version == 0 or self.version == 4:
            self.adjusted_history.append(gradients=adjusted_grad)
        elif self.version == 1:
            self.adjusted_history.append(losses=adjusted_loss,
                                         gradients=adjusted_grad)
        elif self.version == 2 or self.version == 3:
            self.adjusted_history.append(weights=adjusted_wght,
                                         losses=adjusted_loss,
                                         gradients=adjusted_grad)
        state = self.adjusted_history.build_multistate()
        states = {MultiOptimize.AGENT_FMT.format(i): list(v) + [a]
                  for i, v, a in enzip(state, self.model.layer_activations)}
        reward = -float(loss)
        terminal = self._terminal()

        accu = None
        if seq_idx == 0 or terminal:
            features = self.sequence.features
            labels = self.sequence.labels
            accu = self.model.compute_accuracy(features, labels)

        info = {
            'loss': loss, 'accuracy': accu,
            'weights_mean': np.mean(np.abs(new_weights)),
            'weights_sum': np.sum(np.abs(new_weights)),
            'actions_mean': np.mean(action),
            'actions_std': np.std(action),
            'states_mean': np.mean(np.abs(state)),
            'states_sum': np.sum(np.abs(state)),
            'grads_mean': np.mean(self.history['gradients']),
            'grads_sum': np.sum(self.history['gradients']),
            'loss_mean': np.mean(self.history['losses']),
            'adjusted_loss': float(adjusted_loss),
            'adjusted_grad': np.mean(np.abs(adjusted_grad)),
            'grad_diff': np.mean(np.abs(past_grads[0] - past_grads[1]))
        }

        return states, reward, terminal, info

    def _terminal(self):
        return self.current_step >= self.max_batches

    def render(self, mode='human'):
        pass

    def close(self):
        pass
