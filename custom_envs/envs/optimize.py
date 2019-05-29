"""The Optimize environment."""

import numpy as np
import numpy.random as npr

from gym.spaces import Box

from custom_envs import load_data
from custom_envs.models import ModelNumpy as Model
from custom_envs.envs import BaseEnvironment

__author__ = "Adolfo Gonzalez III <adolfo.gonzalez02@utrgv.edu>"


class Optimize(BaseEnvironment):

    """
    Summary:
    The optimize environment requires an agent to reduce the
    objective function of a target neural network on some dataset by changing
    the weights. The action of the agent is added to the current weights of
    the target neural network.

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

    metadata = {
        'render.modes': []
    }

    def __init__(self, data_set='mnist', batch_size=None, n_of_steps=None):
        self.sequence = load_data(data_set, batch_size)
        num_of_labels = self.sequence.label_shape[0]
        feature_size = self.sequence.feature_shape[0]
        self.model = Model(feature_size, num_of_labels)
        past_history = 3
        self.loss_hist = np.zeros(past_history)
        self.grad_hist = np.zeros((past_history, feature_size,
                                   num_of_labels))
        self.wght_hist = np.zeros(self.grad_hist.shape)
        state_size = 2*self.model.size + 1
        self.observation_space = Box(low=-1e3, high=1e3, dtype=np.float32,
                                     shape=(state_size,))
        self.action_space = Box(low=-1e3, high=1e3, dtype=np.float32,
                                shape=(self.model.size,))
        self.seed()

    def base_reset(self):
        #shuffle(self.features, self.labels, np_random=self.np_random)
        self.loss_hist.fill(0)
        self.grad_hist.fill(0)
        self.wght_hist.fill(0)
        self.model.reset(npr)
        return np.concatenate([self.wght_hist[0].ravel(), self.loss_hist[[0]],
                               self.grad_hist[0].ravel()])

    def base_step(self, a):
        idx = self.current_step % len(self.loss_hist)

        features, labels = self.sequence[0]
        self.model.set_weights(self.model.weights -
                               a.reshape((-1, self.wght_hist.shape[-1])))
        loss, grad, accu = self.model.compute_backprop(features, labels)

        grad = grad / len(features)

        np.divide(loss - self.loss_hist[idx-1],
                  self.loss_hist[idx-1] + 0.1, out=self.loss_hist[[idx]])
        np.divide(grad, np.abs(self.grad_hist[idx-1]) + 1,
                  out=self.grad_hist[idx])
        np.divide(np.abs(self.wght_hist[idx-1] - self.wght_hist[idx-2]),
                  np.abs(self.wght_hist[idx] - self.wght_hist[idx-1]) + 0.1,
                  out=self.wght_hist[idx])

        state = np.concatenate([self.wght_hist[idx].ravel(),
                                self.loss_hist[[idx]],
                                self.grad_hist[idx].ravel()])
        reward = -loss
        info = {'objective': loss, 'accuracy': accu}
        return state, reward, self._terminal(), info

    def _terminal(self):
        return self.current_step >= 40

    def render(self, mode='human'):
        pass

    def close(self):
        pass


def __main():
    from stable_baselines.ppo2 import PPO2
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    from stable_baselines.common.vec_env import DummyVecEnv
    env = DummyVecEnv([Optimize])
    agent = PPO2(MlpPolicy, env, verbose=1)
    agent.learn(total_timesteps=10**2)


if __name__ == '__main__':
    __main()
