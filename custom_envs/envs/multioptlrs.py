"""The Optimize environment."""

import numpy as np
import numpy.random as npr

from gym.spaces import Box

from custom_envs import load_data
from custom_envs.models import ModelNumpy as Model
#from custom_envs.models import ModelKeras as Model
from custom_envs.envs import BaseMultiEnvironment

# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py


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

    AGENT_FMT = 'parameter-{:d}'

    def __init__(self, data_set='iris', batch_size=None, version=1):
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
            state_size = 1
        elif version == 1:
            state_size = 2
        elif version == 2:
            state_size = 2 + 1
        elif version == 3:
            state_size = 2*past_history
        self.observation_spaces = {MultiOptLRs.AGENT_FMT.format(i):
                                   Box(low=-1e3, high=1e3,
                                       dtype=np.float32,
                                       shape=(state_size,))
                                   for i in range(self.model.size)}
        self.action_spaces = {MultiOptLRs.AGENT_FMT.format(i):
                              Box(low=-1e3, high=1e3,
                                  dtype=np.float32,
                                  shape=(1,))
                              for i in range(self.model.size)}
        self.seed()
        self.adjusted_loss_hist = np.zeros(self.loss_hist.shape)
        self.adjusted_grad_hist = np.zeros(self.grad_hist.shape)
        self.adjusted_wght_hist = np.zeros(self.wght_hist.shape)
        self.version = version

    def base_reset(self):
        self.loss_hist.fill(0)
        self.grad_hist.fill(0)
        self.wght_hist.fill(0)
        self.adjusted_loss_hist.fill(0)
        self.adjusted_grad_hist.fill(0)
        self.adjusted_wght_hist.fill(0)
        self.model.reset(npr)
        self.sequence.shuffle()
        #self.sequence.features = np.roll(self.sequence.features, 1, axis=1)
        if self.version == 0:
            grad_flat = self.grad_hist[0].ravel()
            states = {MultiOptLRs.AGENT_FMT.format(i): gd
                      for i, gd in enumerate(grad_flat)}
        elif self.version == 1:
            loss_flat = float(self.loss_hist[0])
            grad_flat = self.grad_hist[0].ravel()
            states = {MultiOptLRs.AGENT_FMT.format(i): (loss_flat, gd)
                      for i, gd in enumerate(grad_flat)}
        elif self.version == 2:
            wght_flat = self.wght_hist[0].ravel()
            loss_flat = float(self.loss_hist[0])
            grad_flat = self.grad_hist[0].ravel()
            states = {MultiOptLRs.AGENT_FMT.format(i): (wght, loss_flat, gd)
                      for i, (wght, gd) in
                      enumerate(zip(wght_flat, grad_flat))}
        elif self.version == 3:
            loss_flat = self.adjusted_loss_hist.ravel()
            grad_flat = self.adjusted_grad_hist.ravel()
            states = {MultiOptLRs.AGENT_FMT.format(i): (loss_flat, gd)
                      for i, gd in enumerate(grad_flat)}
        return states

    def base_step(self, actions):
        shape = (-1, self.wght_hist.shape[-1])
        action = np.reshape([actions['parameter-{:d}'.format(i)]
                             for i in range(self.model.size)], shape)
        idx = self.current_step % len(self.loss_hist)

        seq_idx = self.current_step % len(self.sequence)
        features, labels = self.sequence[seq_idx]
        loss, grad, accu = self.model.compute_backprop(features, labels)
        self.model.set_weights(self.model.weights - grad*action)
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
            abs_wght = np.abs(
                self.wght_hist[idx] - self.wght_hist[idx-1]) + 0.1
            adjusted_wght = np.divide(prev_wght, abs_wght)
        else:
            adjusted_loss = np.divide(loss - self.loss_hist[idx-1],
                                      np.abs(self.loss_hist[idx-1]) + 0.1)
            adjusted_grad = np.divide(grad, np.abs(self.grad_hist[idx-1]) + 1)
            adjusted_grad = np.divide(grad, np.abs(self.grad_hist[idx-1]) + 1)
            prev_wght = np.abs(self.wght_hist[idx-1] - self.wght_hist[idx-2])
            abs_wght = np.abs(
                self.wght_hist[idx] - self.wght_hist[idx-1]) + 0.1
            adjusted_wght = np.divide(prev_wght, abs_wght)

        self.adjusted_loss_hist[idx] = adjusted_loss
        self.adjusted_grad_hist[idx] = adjusted_grad
        self.adjusted_wght_hist[idx] = adjusted_wght

        if self.version == 0:
            grad_flat = adjusted_grad.ravel()
            states = {MultiOptLRs.AGENT_FMT.format(i): gd
                      for i, gd in enumerate(grad_flat)}
        elif self.version == 1:
            loss_flat = float(adjusted_loss)
            grad_flat = adjusted_grad.ravel()
            states = {MultiOptLRs.AGENT_FMT.format(i): (loss_flat, gd)
                      for i, gd in enumerate(grad_flat)}
        elif self.version == 2:
            wght_flat = adjusted_wght.ravel()
            loss_flat = float(adjusted_loss)
            grad_flat = adjusted_grad.ravel()
            states = {MultiOptLRs.AGENT_FMT.format(i): (wght, loss_flat, gd)
                      for i, (wght, gd) in
                      enumerate(zip(wght_flat, grad_flat))}
        elif self.version == 3:
            loss_flat = self.adjusted_loss_hist.ravel()
            grad_flat = self.adjusted_grad_hist.ravel()
            states = {MultiOptLRs.AGENT_FMT.format(i): (loss_flat, gd)
                      for i, gd in enumerate(grad_flat)}
        reward = -float(adjusted_loss)
        rewards = {MultiOptLRs.AGENT_FMT.format(i): reward
                   for i in range(self.model.size)}
        terminal = self._terminal()
        terminals = {MultiOptLRs.AGENT_FMT.format(i): terminal
                     for i in range(self.model.size)}
        if terminal:
            features = self.sequence.features
            labels = self.sequence.labels
            loss, _, accu = self.model.compute_backprop(features, labels)
        info = {'objective': loss, 'accuracy': accu}
        infos = {MultiOptLRs.AGENT_FMT.format(i): info
                 for i in range(self.model.size)}

        return states, rewards, terminals, infos

    def _terminal(self):
        return self.current_step >= 400

    def render(self, mode='human'):
        pass

    def close(self):
        pass
