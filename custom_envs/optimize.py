"""classic Acrobot task"""

import numpy as np
import numpy.random as npr
import numexpr
import cv2 as cv

from gym import core, spaces
from gym.utils import seeding

__author__ = "Adolfo Gonzalez III <adolfo.gonzalez02@utrgv.edu>"

from custom_envs.load_data import load_data
from custom_envs.logger import Logger
from custom_envs.utils import shuffle, to_onehot, softmax, cross_entropy
from custom_envs.model import Model

LOG_EVERY = 1000
# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py

class Optimize(core.Env):

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

    def __init__(self):
        self.env_id = npr.randint(0, 1e3)
        data = load_data()
        samples, labels = np.hsplit(data, [-1])
        #samples = samples.reshape((-1, 28, 28, 1))
        #samples = [cv.resize(s, (14, 14)).ravel() for s in samples]
        #samples = np.stack(samples)
        self.samples = samples
        self.features = samples
        self.labels, num_of_labels = to_onehot(labels)
        self.feature_size = samples.shape[-1]
        self.model = Model(self.feature_size, num_of_labels)
        self.H = 3
        self.obj_list = np.zeros(self.H)
        self.grad_list = np.zeros((self.H, self.feature_size, num_of_labels))
        self.W_list = np.zeros_like(self.grad_list)
        self.w = npr.normal(size=(self.feature_size, num_of_labels))
        #self.N = self.obj_list.size + self.grad_list.size + self.w.size 
        self.N = 2*(self.feature_size*num_of_labels) + 1
        print("N: ", self.N)
        self.observation_space = spaces.Box(low=-1e8, high=1e8, 
                                            shape=(self.N,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1e8, high=1e8,
                                       shape=(self.w.size,), dtype=np.float32)
        self.steps = 1
        self.seed()
        self.rewards = []
        self.running_rewards = []
        self.running_acc = []
        self.num_of_labels = num_of_labels
        self.logger = None
        self.global_steps = 0
        self.seed_no = 0

    def seed(self, seed=None):
        self.seed_no = seed
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.logger = Logger('OptDist-v0:iris', self.seed_no)
        if self.rewards:
            #sw = self.samples @ self.w
            #prediction = sigmoid(sw)
            #prediction = softmax(sw)
            #prediction = np.argmax(prediction, axis=1)
            #labels = np.argmax(self.labels, axis=1)
            #accuracy = np.mean(prediction == labels)
            accuracy = self.model.compute_accuracy(self.features, self.labels)
            self.running_acc.append(accuracy)
            self.running_rewards.append(np.sum(self.rewards))
        if len(self.running_rewards) >= LOG_EVERY:
            avg_rew = np.mean(self.running_rewards)
            avg_acc = np.mean(self.running_acc)
            print(np.mean(self.w))
            print(('Mean Reward {:+9.6f} and Mean Accuracy {:3.2f} '
                   'for past {:3d} Exp.').format(avg_rew, avg_acc, LOG_EVERY))
            self.running_rewards = []
            self.running_acc = []
            self.logger.write(self.global_steps, avg_rew, avg_acc,
                              np.max(np.abs(self.w)))
        shuffle(self.samples, self.labels, np_random=self.np_random)
        self.obj_list.fill(0)
        self.grad_list.fill(0)
        self.steps = 1
        self.rewards = []
        self.w = self.np_random.normal(size=self.w.shape)
        #self.w.fill(0)
        self.W_list.fill(0)
        I = 0
        self.model.reset(self.np_random)
        return np.concatenate([self.W_list[I].ravel(), self.obj_list[[I]],
                                self.grad_list[I].ravel()])

    def step(self, a):
        self.global_steps += 1
        #batch_size = 128
        I = self.steps % self.H
        #old_w = self.w.copy()
        #self.w = self.w - a.reshape((-1, self.num_of_labels))
        labels = self.labels#[batch_size*I:batch_size*(I+1)]
        samples = self.samples#[batch_size*I:batch_size*(I+1)]
        features = samples
        #sw = samples @ self.w
        #prediction = sigmoid(sw)
        #prediction = softmax(sw)
        #objective = mse(prediction, labels)
        #objective = cross_entropy(prediction, labels)
        #gradient = numexpr.evaluate('(-prediction*(1-prediction)) * (labels - prediction)')
        #gradient = (samples.T @ gradient)
        
        
        self.model.set_weights(self.model.weights + a.reshape((-1, self.num_of_labels)))
        objective, gradient, _ = self.model.compute_backprop(features, labels)

        self.obj_list[I] = (objective - self.obj_list[I-1]) / (self.obj_list[I-1] + 0.1)
        self.grad_list[I] = gradient / (np.abs(self.grad_list[I-1])+1)
        self.W_list[I] = np.abs(self.W_list[I-1] - self.W_list[I-2])/(np.abs(self.W_list[I] - self.W_list[I-1]) + 0.1)
        self.steps += 1
        
        state = np.concatenate([self.W_list[I].ravel(), self.obj_list[[I]],
                                self.grad_list[I].ravel()])
        terminal = self._terminal()
        #sw = self.samples @ self.w
        #prediction = softmax(sw)
        #print(prediction[:20])
        #prediction = sigmoid(sw)
        #objective = mse(prediction, self.labels)
        #objective = cross_entropy(prediction, self.labels)
        reward = -objective
        #print(reward)
        self.rewards.append(reward)
        accuracy = None
        if terminal:
            accuracy = self.model.compute_accuracy(features, labels)
        return state, reward, terminal, {'objective': objective, 'accuracy': accuracy}

    def _terminal(self):
        return self.steps >= 40

    def render(self, mode='human'):
        pass

    def close(self):
        self.logger.close()


if __name__ == '__main__':
    from stable_baselines.ppo2 import PPO2
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    from stable_baselines.common.vec_env import DummyVecEnv
    env = DummyVecEnv([Optimize])
    agent = PPO2(MlpPolicy, env)
    agent.learn(total_timesteps=2**20)
