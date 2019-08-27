"""The MultiOptimize environment."""

from collections import namedtuple

import numpy as np
from gym.spaces import Box, Dict

import custom_envs.utils.utils_env as utils_env
from custom_envs import load_data
from custom_envs.problems import get_problem
from custom_envs.envs import BaseMultiEnvironment
from custom_envs.utils.utils_common import History, enzip


VersionType = namedtuple('VersionType', ['history', 'observation', 'action',
                                         'reward'])


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
                 max_batches=400, max_history=5, observation_version=0,
                 action_version=0, reward_version=0):
        super().__init__()
        self.model = get_problem(data_set=load_data(data_set, batch_size))
        model_size = (self.model.size,)
        self.history = History(
            5, losses=(), gradients=model_size, weights=model_size
        )
        result = utils_env.get_obs_version(model_size, max_history, version)
        obs_space, self.adjusted_history = result
        if action_version == 0:
            action_low = -4.
            action_high = 4.
        elif action_version == 1:
            action_low = -1e8
            action_high = 1e8
        else:
            raise RuntimeError()

        self.observation_space = Dict({
            MultiOptimize.AGENT_FMT.format(i): obs_space
            for i in range(self.model.size)
        })
        self.action_space = Dict({
            MultiOptimize.AGENT_FMT.format(i):
            Box(low=action_low, high=action_high,
                dtype=np.float32,
                shape=(1,))
            for i in range(self.model.size)
        })
        self.seed()
        self.max_history = max_history
        self.max_batches = max_batches
        self.version = VersionType(
            version, observation_version, action_version, reward_version
        )

    def base_reset(self):
        self.adjusted_history.reset()
        self.model.reset()
        grad, loss, weight = self.model.get()
        self.history.append(losses=loss, gradients=grad, weights=weight)
        states = self.adjusted_history.build_multistate()
        states = {
            MultiOptimize.AGENT_FMT.format(i): list(v)
            for i, v in enumerate(states)
        }
        return states

    def base_step(self, action):
        action = np.reshape([
            action[MultiOptimize.AGENT_FMT.format(i)]
            for i in range(self.model.size)
        ], (-1,))
        if self.version.action == 0:
            sign = np.sign(action)
            mag = np.abs(action) - 3
            action = sign*10**mag
        elif self.version.action == 1:
            action = action*1e-3
        else:
            raise RuntimeError()
        self.model.set_parameters(self.model.parameters - action)
        grad, loss, new_weights = self.model.get()

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
            MultiOptimize.AGENT_FMT.format(i): list(v)
            for i, v in enumerate(state)
        }
        reward = utils_env.get_reward(loss, adjusted_loss, self.version.reward)
        terminal = self._terminal()

        data_loss = None
        if terminal:
            data_loss = self.model.get_loss()
        past_grads = self.history['gradients']
        info = {
            'loss': data_loss,
            'batch_loss': loss,
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
