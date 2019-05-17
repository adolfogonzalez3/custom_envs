"""classic Acrobot task"""
"""NOT SUCCESSFUL"""

import numpy as np
import numpy.random as npr
import numexpr
import cv2 as cv

from gym import core, spaces
from gym.utils import seeding

import mnist

from stable_baselines.ppo2 import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv
__author__ = "Adolfo Gonzalez III <adolfo.gonzalez02@utrgv.edu>"

from custom_envs.load_data import load_data
from custom_envs.logger import Logger

LOG_EVERY = 1000

# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py

def shuffle(*args, np_random=npr):
    length = len(args[0])
    indices = np.arange(length)
    npr.shuffle(indices)
    return [arg[indices] for arg in args]

def cross_entropy(p, y):
    #p_log = np.nan_to_num(np.log(p))
    p_log = np.log(p+1e-16)
    return np.mean(np.sum(-p_log*y, axis=1))
    
def mse(p, y):
    return np.mean(np.sum((p - y)**2, axis=1)/2)

def softmax(x):
    x = (x - np.max(x))/(np.max(x) - np.min(x) + 1e-8)
    #p_exp = np.exp(x - np.max(x))
    p_exp = np.exp(x)
    p_sum = np.sum(p_exp, axis=1)
    return p_exp/p_sum[:, None]

def sigmoid(x):
    return numexpr.evaluate('1 / (1 + exp(-x - 1e-8))')

def to_onehot(x):
    x = x.astype(np.int).ravel()
    num_of_labels = np.unique(x).size
    onehot = np.zeros((len(x), num_of_labels))
    onehot[np.arange(len(x)), x] = 1
    return onehot, num_of_labels

def compute_angle(A, B):
    A_norm = np.sqrt(np.sum(A**2))
    B_norm = np.sqrt(np.sum(B**2))
    angle = np.arccos(np.dot(A.ravel(), B.ravel()) / (A_norm*B_norm))
    return angle

def compute_dist(A, B):
    return np.sqrt(np.sum((A-B)**2))

def standarize(x):
    low = np.min(x)
    high = np.max(x)
    return (x - low) / (high - low + 1e-8)

class OptDist(core.Env):

    """
    The optimize environment requires an agent to reduce the 
    objective function of a neural network on some dataset.
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
        self.labels, num_of_labels = to_onehot(labels)
        self.feature_size = samples.shape[-1]
        self.H = 3
        self.obj_list = np.zeros(self.H)
        self.wdist_list = np.zeros(self.H)
        self.gdist_list = np.zeros(self.H)
        self.wangle_list = np.zeros(self.H)
        self.gangle_list = np.zeros(self.H)
        self.grad_list = np.zeros((self.H, self.feature_size, num_of_labels))
        self.W_list = np.zeros_like(self.grad_list)
        self.w = npr.normal(size=(self.feature_size, num_of_labels))
        #self.N = self.W_list.size + self.grad_list.size + self.w.size 
        self.N = self.w.size + 2*self.H + 1
        self.observation_space = spaces.Box(low=-1e8, high=1e8, 
                                            shape=(self.N,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1e8, high=1e8,
                                       shape=(self.w.size,), dtype=np.float32)
        self.steps = 1
        self.seed()
        self.seed_no = None
        self.rewards = []
        self.running_rewards = []
        self.running_acc = []
        self.num_of_labels = num_of_labels
        self.logger = None
        self.global_steps = 0

    def seed(self, seed=None):
        self.seed_no = seed
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.logger = Logger('OptDist-v0:iris', self.seed_no)
        if self.rewards:
            sw = self.samples @ self.w
            #prediction = sigmoid(sw)
            prediction = softmax(sw)
            prediction = np.argmax(prediction, axis=1)
            labels = np.argmax(self.labels, axis=1)
            self.running_acc.append(np.mean(prediction == labels))
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
        self.obj_list.fill(0)
        self.grad_list.fill(0)
        self.gdist_list.fill(0)
        self.gangle_list.fill(0)
        self.steps = 1
        self.rewards = []
        self.w = self.np_random.normal(size=self.w.shape)
        #self.w.fill(0)
        self.W_list.fill(0)
        I = 0
        return np.concatenate([standarize(self.W_list[I].ravel()),
                               self.obj_list[[I]],
                               self.gdist_list, self.gangle_list])

    def step(self, a):
        self.global_steps += 1
        #batch_size = 128
        I = self.steps % self.H
        old_w = self.w.copy()
        self.w = self.w - a.reshape((-1, self.num_of_labels))
        labels = self.labels#[batch_size*I:batch_size*(I+1)]
        samples = self.samples#[batch_size*I:batch_size*(I+1)]
        sw = samples @ self.w
        #prediction = sigmoid(sw)
        prediction = softmax(sw)
        #objective = mse(prediction, labels)
        objective = cross_entropy(prediction, labels)
        gradient = numexpr.evaluate('(-prediction*(1-prediction)) * (labels - prediction)')
        gradient = (samples.T @ gradient)
        
        
        
        self.obj_list[I] = (objective - self.obj_list[I-1]) / (self.obj_list[I-1] + 0.1)
        self.grad_list[I] = gradient # / (np.abs(self.grad_list[I-1])+1)
        
        self.gdist_list[I] = np.sqrt(np.sum(self.grad_list[I]**2))
        self.gangle_list[I] = compute_angle(self.grad_list[I], self.w)
        
        

        
        
        self.obj_list[I] = (objective - self.obj_list[I-1]) / (self.obj_list[I-1] + 0.1)
        self.grad_list[I] = gradient / (np.abs(self.grad_list[I-1])+1)
        self.W_list[I] = np.abs(self.W_list[I-1] - self.W_list[I-2])/(np.abs(self.W_list[I] - self.W_list[I-1]) + 0.1)
        self.steps += 1
        
        state = np.concatenate([self.W_list[I].ravel(), self.obj_list[[I]],
                                self.gdist_list, self.gangle_list])
        terminal = self._terminal()
        sw = self.samples @ self.w
        prediction = softmax(sw)
        #print(prediction[:20])
        #prediction = sigmoid(sw)
        #objective = mse(prediction, self.labels)
        objective = cross_entropy(prediction, self.labels)
        reward = -objective
        #print(reward)
        self.rewards.append(reward)
        if np.any(np.isnan(state)):
            print(state)
            print("NAN DETECTED")
            exit()
        return state, reward, terminal, {}

    def _terminal(self):
        return self.steps >= 40


    def _get_state(self):
        pass


    def render(self, mode='human'):
        pass

    def close(self):
        pass


if __name__ == '__main__':
    env = DummyVecEnv([OptDist])
    agent = PPO2(MlpPolicy, env)
    agent.learn(total_timesteps=10**7)
