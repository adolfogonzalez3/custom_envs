"""The Optimize environment."""

from collections import namedtuple

import numpy as np
from gym.spaces import Box, Dict

import custom_envs.utils.utils_env as utils_env
from custom_envs import load_data
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
        self.history = History(5, losses=(), gradients=(self.model.size,),
                               weights=(self.model.size,))
        result = utils_env.get_obs_version((self.model.size,), max_history,
                                           version)
        obs_space, self.adjusted_history = result
        act_space = utils_env.get_action_space_optlrs(action_version)

        self.observation_space = Dict({
            MultiOptLRs.AGENT_FMT.format(i): obs_space
            for i in range(self.model.size)
        })
        self.action_space = Dict({
            MultiOptLRs.AGENT_FMT.format(i): act_space
            for i in range(self.model.size)
        })
        self.seed()
        self.max_history = max_history
        self.max_batches = max_batches
        self.version = VersionType(version, observation_version,
                                   action_version, reward_version)

    def base_reset(self):
        self.adjusted_history.reset()
        self.model.reset()
        if True:
            seq_idx = len(self.sequence) - 1
            features, labels = self.sequence[seq_idx]
            loss, grad, _ = self.model.compute_backprop(features, labels)
            self.history.reset(losses=loss, gradients=grad,
                               weights=self.model.weights)
        else:
            self.history.reset()
        states = self.adjusted_history.build_multistate()
        states = {
            MultiOptLRs.AGENT_FMT.format(i): list(v)
            for i, v, a in enzip(states, self.model.layer_activations)
        }
        return states

    def base_step(self, action):
        action = np.reshape([action[MultiOptLRs.AGENT_FMT.format(i)].ravel()
                             for i in range(self.model.size)], (-1,))
        seq_idx = self.current_step % len(self.sequence)
        features, labels = self.sequence[seq_idx]

        loss, grad, accu = self.model.compute_backprop(features, labels)
        action = utils_env.get_action_optlrs(action, self.version.action)
        new_weights = self.model.weights - grad*action

        self.model.set_weights(new_weights)
        loss = self.model.compute_loss(features, labels)
        seq_idx = (self.current_step + 1) % len(self.sequence)
        if seq_idx == 0:
            self.sequence.shuffle()
        features, labels = self.sequence[seq_idx]
        _, grad, accu = self.model.compute_backprop(features, labels)

        self.history.append(losses=loss, gradients=grad, weights=new_weights)
        adjusted = utils_env.get_observation(self.history,
                                             self.version.observation)
        adjusted_loss, adjusted_wght, adjusted_grad = adjusted

        if self.version.history == 0 or self.version.history == 4:
            self.adjusted_history.append(gradients=adjusted_grad)
        elif self.version.history == 1:
            self.adjusted_history.append(losses=adjusted_loss,
                                         gradients=adjusted_grad)
        elif self.version.history == 2 or self.version.history == 3:
            self.adjusted_history.append(weights=adjusted_wght,
                                         losses=adjusted_loss,
                                         gradients=adjusted_grad)
        elif self.version.history == 5:
            self.adjusted_history.append(losses=adjusted_loss,
                                         gradients=adjusted_grad,
                                         actions=action)
        else:
            raise RuntimeError()
        state = self.adjusted_history.build_multistate()
        states = {
            MultiOptLRs.AGENT_FMT.format(i): list(v)
            for i, v, a in enzip(state, self.model.layer_activations)
        }
        reward = utils_env.get_reward(loss, adjusted_loss, self.version.reward)
        terminal = self._terminal()

        accuracy = None
        data_loss = None
        if seq_idx == 0 or terminal:
            features = self.sequence.features
            labels = self.sequence.labels
            accuracy = self.model.compute_accuracy(features, labels)
            data_loss = self.model.compute_loss(features, labels)
        past_grads = self.history['gradients']
        info = {
            'loss': data_loss, 'accuracy': accuracy,
            'batch_loss': loss, 'batch_accuracy': accu,
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
