"""The Optimize environment."""

import numpy as np
import numpy.random as npr

from gym.spaces import Box

from custom_envs import load_data
from custom_envs.models import ModelNumpy as Model
#from custom_envs.models import ModelKeras as Model
from custom_envs.envs import BaseEnvironment

__author__ = "Adolfo Gonzalez III <adolfo.gonzalez02@utrgv.edu>"


class OptimizeCorrect(BaseEnvironment):

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

    def __init__(self, data_set='mnist', batch_size=None, version=1):
        super().__init__()
        self.sequence = load_data(data_set, batch_size)
        num_of_labels = self.sequence.label_shape[0]
        feature_size = self.sequence.feature_shape[0]
        self.model = Model(feature_size, num_of_labels)
        past_history = 5
        self.loss_hist = np.zeros((past_history, 1))
        self.grad_hist = np.zeros((past_history, feature_size,
                                   num_of_labels))
        self.wght_hist = np.zeros(self.grad_hist.shape)
        if version == 0:
            state_size = self.model.size
        elif version == 1:
            state_size = self.model.size + 1
        elif version == 2:
            state_size = 2*self.model.size + 1
        elif version == 3:
            state_size = (self.model.size + 1)*past_history
        #state_size = 2*self.model.size + 1
        #state_size = self.loss_hist.size + self.grad_hist.size + self.wght_hist.size
        self.observation_space = Box(low=-1e3, high=1e3, dtype=np.float32,
                                     shape=(state_size,))
        #self.action_space = Box(low=-1e3, high=1e3, dtype=np.float32,
        #                        shape=(self.model.size,))
        self.action_space = Box(low=0, high=1, dtype=np.float32,
                                shape=(self.model.size,))
        self.seed()
        self.adjusted_loss_hist = np.zeros(self.loss_hist.shape)
        self.adjusted_grad_hist = np.zeros(self.grad_hist.shape)
        self.adjusted_wght_hist = np.zeros(self.wght_hist.shape)
        self.version = version

    def base_reset(self):
        #shuffle(self.features, self.labels, np_random=self.np_random)
        self.loss_hist.fill(0)
        self.grad_hist.fill(0)
        self.wght_hist.fill(0)
        self.adjusted_loss_hist.fill(0)
        self.adjusted_grad_hist.fill(0)
        self.adjusted_wght_hist.fill(0)
        self.model.reset(npr)
        self.sequence.shuffle()
        self.sequence.features = np.roll(self.sequence.features, 1, axis=1)
        if self.version == 0:
            return self.grad_hist[0].ravel()
        elif self.version == 1:
            return np.concatenate([self.loss_hist[0].ravel(),
                                   self.grad_hist[0].ravel()])
        elif self.version == 2:
            return np.concatenate([self.wght_hist[0].ravel(),
                                   self.loss_hist[0].ravel(),
                                   self.grad_hist[0].ravel()])
        elif self.version == 3:
            return np.concatenate([self.adjusted_loss_hist.ravel(),
                                   self.adjusted_grad_hist.ravel()])

    def base_step(self, a):
        idx = self.current_step % len(self.loss_hist)

        seq_idx = self.current_step % len(self.sequence)
        features, labels = self.sequence[0]
        loss, grad, accu = self.model.compute_backprop(features, labels)
        self.model.set_weights(self.model.weights -
                               grad*a.reshape((-1, self.wght_hist.shape[-1])))
        loss, grad, accu = self.model.compute_backprop(features, labels)

        grad = grad / len(features)
        self.loss_hist[idx] = loss
        self.grad_hist[idx] = grad
        self.wght_hist[idx] = self.model.weights

        if self.version == 3:
            mean_loss = np.mean(self.loss_hist)
            adjusted_loss = np.divide(loss - mean_loss, np.abs(mean_loss) + .1)
            mean_grad = np.mean(self.grad_hist)
            adjusted_grad = np.divide(grad, np.abs(mean_grad) + 1)
            prev_wght = np.abs(self.wght_hist[idx-1] - self.wght_hist[idx-2])
            abs_wght = np.abs(self.wght_hist[idx] - self.wght_hist[idx-1]) + 0.1
            adjusted_wght = np.divide(prev_wght, abs_wght)
        else:
            adjusted_loss = np.divide(loss - self.loss_hist[idx-1],
                                      np.abs(self.loss_hist[idx-1]) + 0.1)
            adjusted_grad = np.divide(grad, np.abs(self.grad_hist[idx-1]) + 1)
            adjusted_grad = np.divide(grad, np.abs(self.grad_hist[idx-1]) + 1)
            prev_wght = np.abs(self.wght_hist[idx-1] - self.wght_hist[idx-2])
            abs_wght = np.abs(self.wght_hist[idx] - self.wght_hist[idx-1]) + 0.1
            adjusted_wght = np.divide(prev_wght, abs_wght)

        self.adjusted_loss_hist[idx] = adjusted_loss
        self.adjusted_grad_hist[idx] = adjusted_grad
        self.adjusted_wght_hist[idx] = adjusted_wght
        
        if self.version == 0:
            state = adjusted_grad.ravel()
        elif self.version == 1:
            state = np.concatenate([adjusted_loss.ravel(),
                                    adjusted_grad.ravel()])
        elif self.version == 2:
            state = np.concatenate([adjusted_wght.ravel(),
                                    adjusted_loss.ravel(),
                                    adjusted_grad.ravel()])
        elif self.version == 3:
            state = np.concatenate([self.adjusted_loss_hist.ravel(),
                                    self.adjusted_grad_hist.ravel()])
            #state = np.zeros_like(state)
        reward = -float(adjusted_loss)
        terminal = self._terminal()
        if terminal:
            features = self.sequence.features
            labels = self.sequence.labels
            loss, _, accu = self.model.compute_backprop(features, labels)
        info = {'objective': loss, 'accuracy': accu}
        #print(reward)
        return state, reward*100, terminal, info

    def _terminal(self):
        return self.current_step >= 400

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
