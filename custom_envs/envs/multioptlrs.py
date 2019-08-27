"""The Optimize environment."""

from collections import namedtuple

import numpy as np
from gym.spaces import Dict

import custom_envs.utils.utils_env as utils_env
from custom_envs import load_data
from custom_envs.problems import get_problem
from custom_envs.envs import BaseMultiEnvironment
from custom_envs.utils.utils_common import History

VersionType = namedtuple('VersionType', ['history', 'observation', 'action',
                                         'reward'])

BOUNDS = 1e2


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

    def __init__(self, problem='func', max_batches=400, max_history=5):
        super().__init__()
        self.model = get_problem(problem)
        self.history = History(
            5, losses=(), gradients=(self.model.size,),
            weights=(self.model.size,)
        )
        result = utils_env.get_obs_version((self.model.size,), max_history, 3)
        obs_space, self.adjusted_history = result
        act_space = utils_env.get_action_space_optlrs(0)

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
        self.version = VersionType(3, 3, 0, 1)

    def base_reset(self):
        self.adjusted_history.reset()
        self.model.reset()
        self.history.reset()
        grad, loss, weight = self.model.get()
        self.history.append(losses=loss, gradients=grad, weights=weight)
        states = self.adjusted_history.build_multistate()
        states = {
            MultiOptLRs.AGENT_FMT.format(i): np.clip(list(v), -BOUNDS, BOUNDS)
            for i, v in enumerate(states)
        }
        return states

    def base_step(self, action):
        action = np.reshape([
            action[MultiOptLRs.AGENT_FMT.format(i)].ravel()
            for i in range(self.model.size)
        ], (-1,))
        grad = self.model.get_gradient()
        action = utils_env.get_action_optlrs(action, self.version.action)
        self.model.set_parameters(self.model.parameters - grad*action)
        grad, loss, weights = self.model.get()
        self.history.append(losses=loss, gradients=grad, weights=weights)
        adj_loss, adj_wght, adj_grad = utils_env.get_observation(
            self.history, self.version.observation
        )
        self.adjusted_history.append(
            weights=adj_wght, losses=adj_loss, gradients=adj_grad
        )
        state = self.adjusted_history.build_multistate()
        states = {
            MultiOptLRs.AGENT_FMT.format(i): np.clip(list(v), -BOUNDS, BOUNDS)
            for i, v in enumerate(state)
        }
        reward = utils_env.get_reward(loss, adj_loss, 1)
        reward = np.clip(reward, -BOUNDS, BOUNDS)
        terminal = self._terminal()
        if not terminal and loss > 1e3:
            terminal = True
        final_loss = None
        if terminal:
            final_loss = self.model.get_loss()
        past_grads = self.history['gradients']
        info = {
            'loss': final_loss,
            'batch_loss': loss,
            'weights_mean': np.mean(np.abs(weights)),
            'weights_sum': np.sum(np.abs(weights)),
            'actions_mean': np.mean(action),
            'actions_std': np.std(action),
            'states_mean': np.mean(np.abs(state)),
            'states_sum': np.sum(np.abs(state)),
            'grads_mean': np.mean(self.history['gradients']),
            'grads_sum': np.sum(self.history['gradients']),
            'loss_mean': np.mean(self.history['losses']),
            'adjusted_loss': float(adj_loss),
            'adjusted_grad': np.mean(np.abs(adj_grad)),
            'grad_diff': np.mean(np.abs(past_grads[0] - past_grads[1])),
            'weights': weights
        }
        self.model.next()
        return states, reward, terminal, info

    def _terminal(self):
        return self.current_step >= self.max_batches

    def render(self, mode='human'):
        pass

    def close(self):
        pass
