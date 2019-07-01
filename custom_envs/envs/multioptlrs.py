"""The Optimize environment."""

from collections import namedtuple

import numpy as np
from gym.spaces import Box, Dict, Discrete

from custom_envs import load_data
#from custom_envs.models import ModelNumpy as Model
from custom_envs.models import ModelKeras as Model
from custom_envs.envs import BaseMultiEnvironment
from custom_envs.utils.utils_common import History, enzip

VersionType = namedtuple('VersionType', ['history', 'observation', 'action',
                                         'reward'])


class MultiOptLRs(BaseMultiEnvironment):

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

    def __init__(self, data_set='iris', batch_size=None, max_batches=400,
                 max_history=5, version=1, observation_version=0,
                 action_version=0, reward_version=0):
        super().__init__()
        self.sequence = load_data(data_set, batch_size)
        num_of_labels = self.sequence.label_shape[0]
        feature_size = self.sequence.feature_shape[0]
        self.model = Model(feature_size, num_of_labels, use_bias=True)
        model_shape = self.model.weights.shape
        self.history = History(5, losses=(), gradients=model_shape,
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
            state_size = max_history

        if action_version == 0:
            action_low = -3.
            action_high = 1.
        elif action_version == 1:
            action_low = 0.
            action_high = 1e4

        self.observation_space = Dict({MultiOptLRs.AGENT_FMT.format(i):
                                       Box(low=-1e6, high=1e6,
                                           dtype=np.float32,
                                           shape=(state_size,))
                                       for i in range(self.model.size)})
        self.action_space = Dict({MultiOptLRs.AGENT_FMT.format(i):
                                  Box(low=action_low, high=action_high,
                                      dtype=np.float32,
                                      shape=(1,))
                                  for i in range(self.model.size)})
        self.seed()
        self.max_history = max_history
        self.max_batches = max_batches
        self.version = VersionType(version, observation_version, action_version,
                                   reward_version)
        self.running_rate = np.ones(model_shape)

    def base_reset(self):
        self.reward = 0
        self.adjusted_history.reset()
        self.model.reset()
        model_shape = self.model.weights.shape
        self.running_rate = np.ones(model_shape)
        self.history.reset()
        states = self.adjusted_history.build_multistate()
        states = {MultiOptLRs.AGENT_FMT.format(i): list(v)
                  for i, v, a in enzip(states, self.model.layer_activations)}
        return states

    def base_step(self, action):
        action = np.reshape([action[MultiOptLRs.AGENT_FMT.format(i)].ravel()
                             for i in range(self.model.size)], (-1,))
        seq_idx = self.current_step % len(self.sequence)
        if seq_idx == 0:
            self.sequence.shuffle()
        features, labels = self.sequence[seq_idx]

        loss, grad, accu = self.model.compute_backprop(features, labels)
        if self.version.action == 0:
            action = 10**action
        elif self.version.action == 1:
            action = action*1e-3
        new_weights = self.model.weights - grad*action

        self.model.set_weights(new_weights)
        loss, grad, accu = self.model.compute_backprop(features, labels)

        self.history.append(losses=loss, gradients=grad, weights=new_weights)
        past_losses = self.history['losses']
        past_grads = self.history['gradients']
        past_weights = self.history['weights']

        prev_grad = np.abs(past_grads[1] - past_grads[2])
        adjusted_grad = np.divide(past_grads[0] - past_grads[1],
                                  prev_grad + 1e-8)
        adjusted_loss = past_losses[0] / (np.abs(past_losses[1]) + 1e0)
        if self.version.observation == 0:
            adjusted_grad = past_grads[0] / (np.abs(past_grads[1]) + 1e-3)
        elif self.version.observation == 1:
            adjusted_grad = grad * 1e2
        elif self.version.observation == 2:
            if self.current_step < 3:
                adjusted_loss = 1
                adjusted_wght = 1
                adjusted_grad = np.ones_like(grad)
            else:
                prev_loss = np.abs(past_losses[1] - past_losses[2])
                adjusted_loss = np.divide(past_losses[0] - past_losses[1],
                                          prev_loss + 1e-3)
                prev_wght = np.abs(past_weights[1] - past_weights[2])
                abs_wght = np.abs(past_weights[0] - past_weights[1])
                adjusted_wght = np.divide(prev_wght, abs_wght + 1e-8)
                prev_grad = np.abs(past_grads[1] - past_grads[2])
                adjusted_grad = np.divide(past_grads[0] - past_grads[1],
                                          prev_grad + 1e-8)

        if self.version.history == 0 or self.version.history == 4:
            self.adjusted_history.append(gradients=adjusted_grad)
        elif self.version.history == 1:
            self.adjusted_history.append(losses=adjusted_loss,
                                         gradients=adjusted_grad)
        elif self.version.history == 2 or self.version.history == 3:
            self.adjusted_history.append(weights=adjusted_wght,
                                         losses=adjusted_loss,
                                         gradients=adjusted_grad)
        state = self.adjusted_history.build_multistate()
        states = {MultiOptLRs.AGENT_FMT.format(i): list(v)
                  for i, v, a in enzip(state, self.model.layer_activations)}
        self.reward -= float(adjusted_loss)
        if self.version.reward == 0:
            reward = -float(adjusted_loss)
        elif self.version.reward == 1:
            reward = float(1 / loss)
        terminal = self._terminal()

        accu = None
        if seq_idx == 0 or terminal:
            features = self.sequence.features
            labels = self.sequence.labels
            accu = self.model.compute_accuracy(features, labels)
            loss = self.model.compute_loss(features, labels)

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
