"""classic Acrobot task"""

import numpy as np

from gym import core, spaces
from gym.utils import seeding

__author__ = "Adolfo Gonzalez III <adolfo.gonzalez02@utrgv.edu>"

from custom_envs import load_data
from custom_envs.utils import shuffle, to_onehot, softmax, cross_entropy
from custom_envs.model import Model

LOG_EVERY = 1000
# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py

class MultiOptimize(core.Env):

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

    def __init__(self, columnwise=True):
        data = load_data()
        features, labels = np.hsplit(data, [-1])
        self.features = features
        self.labels, num_of_labels = to_onehot(labels)
        self.feature_size = features.shape[-1]
        self.model = Model(self.feature_size, num_of_labels)
        self.H = 3
        self.obj_list = np.zeros(self.H)
        self.grad_list = np.zeros((self.H, self.feature_size, num_of_labels))
        self.W_list = np.zeros_like(self.grad_list)
        # self.N = self.obj_list.size + self.grad_list.size + self.w.size
        self.N = 2*self.model.size + 1
        self.observation_space = spaces.Box(low=-1e8, high=1e8, 
                                            shape=(self.N,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1e8, high=1e8,
                                       shape=(self.model.size,),
                                       dtype=np.float32)
        self.steps = 1
        self.seed()
        self.rewards = []
        self.running_rewards = []
        self.running_acc = []
        self.num_of_labels = num_of_labels
        self.global_steps = 0
        self.seed_no = 0

    def seed(self, seed=None):
        self.seed_no = seed
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        if self.rewards:
            accuracy = self.model.compute_accuracy(self.features, self.labels)
            self.running_acc.append(accuracy)
            self.running_rewards.append(np.sum(self.rewards))
        if len(self.running_rewards) >= LOG_EVERY:
            avg_rew = np.mean(self.running_rewards)
            avg_acc = np.mean(self.running_acc)
            print(('Mean Reward {:+9.6f} and Mean Accuracy {:3.2f} '
                   'for past {:3d} Exp.').format(avg_rew, avg_acc, LOG_EVERY))
            self.running_rewards = []
            self.running_acc = []
        shuffle(self.features, self.labels, np_random=self.np_random)
        self.obj_list.fill(0)
        self.grad_list.fill(0)
        self.W_list.fill(0)
        self.steps = 1
        self.rewards = []
        I = 0
        self.model.reset(self.np_random)
        return np.concatenate([self.W_list[I].ravel(), self.obj_list[[I]],
                               self.grad_list[I].ravel()])

    def step(self, a):
        self.global_steps += 1
        I = self.steps % self.H
        labels = self.labels
        features = self.features

        self.model.set_weights(self.model.weights - a.reshape((-1, self.num_of_labels)))
        objective, gradient, accuracy = self.model.compute_backprop(features, labels)

        self.obj_list[I] = (objective - self.obj_list[I-1]) / (self.obj_list[I-1] + 0.1)
        self.grad_list[I] = gradient / (np.abs(self.grad_list[I-1])+1)
        self.W_list[I] = np.abs(self.W_list[I-1] - self.W_list[I-2])/(np.abs(self.W_list[I] - self.W_list[I-1]) + 0.1)
        self.steps += 1

        state = np.concatenate([self.W_list[I].ravel(), self.obj_list[[I]],
                                self.grad_list[I].ravel()])
        terminal = self._terminal()
        reward = -objective
        self.rewards.append(reward)
        return state, reward, terminal, {'objective': objective, 'accuracy': accuracy}

    def _terminal(self):
        return self.steps >= 40

    def render(self, mode='human'):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    from stable_baselines.ppo2 import PPO2
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    from stable_baselines.common.vec_env import DummyVecEnv
    env = DummyVecEnv([Optimize])
    agent = PPO2(MlpPolicy, env)
    agent.learn(total_timesteps=2**20)
