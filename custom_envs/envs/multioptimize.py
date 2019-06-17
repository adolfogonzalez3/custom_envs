"""The MultiOptimize environment."""

import numpy as np
import numpy.random as npr
from gym.spaces import Box, Dict

from custom_envs import load_data
from custom_envs.models import ModelNumpy as Model
from custom_envs.envs import BaseMultiEnvironment
from custom_envs.utils.utils_common import History

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

    AGENT_FMT = 'parameter-{:d}'

    def __init__(self, data_set='iris', batch_size=None, version=1,
                 max_batches=400, max_history=5):
        super().__init__()
        self.sequence = load_data(data_set, batch_size)
        num_of_labels = self.sequence.label_shape[0]
        feature_size = self.sequence.feature_shape[0]
        model_shape = (feature_size, num_of_labels)
        self.model = Model(feature_size, num_of_labels, use_bias=True)
        model_shape = self.model.weights.shape
        self.history = History(max_history, losses=(), gradients=model_shape,
                               weights=model_shape)
        if version == 0:
            self.adjusted_history = History(1, gradients=model_shape)
            state_size = 1
        elif version == 1:
            self.adjusted_history = History(1, losses=(),
                                            gradients=model_shape)
            state_size = 2
        elif version == 2:
            self.adjusted_history = History(1, weights=model_shape,
                                            losses=(), gradients=model_shape)
            state_size = 3
        elif version == 3:
            self.adjusted_history = History(max_history, weights=model_shape,
                                            losses=(), gradients=model_shape)
            state_size = 3*max_history
        self.observation_spaces = Dict({MultiOptimize.AGENT_FMT.format(i):
                                        Box(low=-1e3, high=1e3,
                                            dtype=np.float32,
                                            shape=(state_size,))
                                        for i in range(self.model.size)})
        self.action_spaces = Dict({MultiOptimize.AGENT_FMT.format(i):
                                   Box(low=-1e3, high=1e3,
                                       dtype=np.float32,
                                       shape=(1,))
                                   for i in range(self.model.size)})
        self.seed()
        self.max_history = max_history
        self.max_batches = max_batches
        self.version = version

    def base_reset(self):
        self.history.reset()
        self.adjusted_history.reset()
        self.model.reset(npr)
        states = self.adjusted_history.build_multistate()
        states = {MultiOptimize.AGENT_FMT.format(i): list(values)
                  for i, values in enumerate(states)}
        return states

    def base_step(self, action):
        shape = (-1, self.history['weights'].shape[-1])
        action = np.reshape([action[MultiOptimize.AGENT_FMT.format(i)]
                             for i in range(self.model.size)], shape)

        seq_idx = self.current_step % len(self.sequence)
        if seq_idx == 0:
            self.sequence.shuffle()
        features, labels = self.sequence[seq_idx]
        self.model.set_weights(self.model.weights - action)
        loss, grad, accu = self.model.compute_backprop(features, labels)

        grad = grad / len(features)
        self.history.append(losses=loss, gradients=grad,
                            weights=self.model.weights)
        past_losses = self.history['losses']
        past_grads = self.history['gradients']
        past_weights = self.history['weights']

        adjusted_loss = np.divide(past_losses[0] - past_losses[1],
                                  np.abs(past_losses[1]) + 0.1)
        adjusted_grad = np.divide(past_grads[0], np.abs(past_grads[1]) + 1)
        prev_wght = np.abs(past_weights[1] - past_weights[2])
        abs_wght = np.abs(past_weights[0] - past_weights[1])
        adjusted_wght = np.divide(prev_wght, abs_wght + 0.1)

        adjusted_loss = np.sign(adjusted_loss)
        adjusted_grad = np.sign(adjusted_grad)
        adjusted_wght = np.sign(adjusted_wght)
        #adjusted_loss = adjusted_loss*1e3
        #adjusted_grad = adjusted_grad*1e3
        #adjusted_wght = adjusted_wght*1e3
        # adjusted_grad = adjusted_grad * 1e3

        if self.version == 0:
            self.adjusted_history.append(gradients=adjusted_grad)
        elif self.version == 1:
            self.adjusted_history.append(losses=adjusted_loss,
                                         gradients=adjusted_grad)
        elif self.version == 2 or self.version == 3:
            self.adjusted_history.append(weights=adjusted_wght,
                                         losses=adjusted_loss,
                                         gradients=adjusted_grad)
        state = self.adjusted_history.build_multistate()
        states = {MultiOptimize.AGENT_FMT.format(i): list(values)
                  for i, values in enumerate(state)}
        reward = -float(adjusted_loss)
        rewards = {MultiOptimize.AGENT_FMT.format(i): reward
                   for i in range(self.model.size)}
        terminal = self._terminal()
        terminals = {MultiOptimize.AGENT_FMT.format(i): terminal
                     for i in range(self.model.size)}
        #if terminal:
        #    features = self.sequence.features
        #    labels = self.sequence.labels
        #    loss, _, accu = self.model.compute_backprop(features, labels)
        info = {'loss': loss, 'accuracy': accu,
                'weights_mean': np.mean(self.model.weights),
                'actions_mean': np.mean(action),
                'states_mean': np.mean(state),
                'grads_mean': np.mean(self.history['gradients']),
                'loss_mean': np.mean(self.history['losses'])
                }
        infos = {MultiOptimize.AGENT_FMT.format(i): info
                 for i in range(self.model.size)}

        return states, rewards, terminals, infos

    def _terminal(self):
        return self.current_step >= self.max_batches

    def render(self, mode='human'):
        pass

    def close(self):
        pass
