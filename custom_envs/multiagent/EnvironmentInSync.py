
from collections import namedtuple
from enum import Enum

from gym import Env
from gym.spaces import Box
import numpy as np

from MailboxInSync import MailboxInSync

class RequestType(Enum):
    STEP = 0
    RESET = 1
    RENDER = 2
    CLOSE = 3


def segment_space(space, segments):
    if issubclass(space, Box):
        n = np.prod(space) / segments
        return [Box(space.low, space.high, shape=(n,), dtype=space.dtype)
                for i in range(segments)]



EnvRequest = namedtuple('EnvRequest', ['type', 'data'])

class EnvSpawn(Env):
    def __init__(self, observation_space, action_space, mailbox):
        self._mailbox = mailbox
        self.action_space = action_space
        self.observation_space = observation_space
        
    def step(self, action):
        request = EnvRequest(RequestType.STEP, action)
        self._mailbox.append(request)
        response = self._mailbox.get()
        return response
    
    def reset(self):
        request = EnvRequest(RequestType.RESET, None)
        self._mailbox.append(request)
        response = self._mailbox.get()
        return response
    
    def render(self):
        request = EnvRequest(RequestType.RENDER, None)
        self._mailbox.append(request)
        
    def close(self):
        request = EnvRequest(RequestType.CLOSE, None)
        self._mailbox.append(request)

    def __del__(self):
        self.close()


class EnvironmentInSync:
    '''A class to use OpenAi algorithms.'''
    
    
    def __init__(self, env, slices=1):
        self.env = env
        self.mailbox = MailboxInSync()
        self.observation_spaces = [env.observation_space]*slices
        self.action_spaces = segment_space(env.action_space, slices)
        self.sub_envs = [EnvSpawn(obs_space, act_space, self.mailbox.spawn())
                         for obs_space, act_space in
                         zip(self.observation_spaces, self.action_spaces)]
        
        
    def handle_requests(self):
        obs = rewards = dones = info = None
        requests = self.mailbox.get()
        if all([r.type == RequestType.RESET for r in requests]):
            obs = self.env.reset()
            self.mailbox.append(obs)
            return self.handle_requests()
        elif all([r.type == RequestType.CLOSE for r in requests]):
            return None
        elif all([r.type == RequestType.STEP for r in requests]):
            action = np.concatenate([r.data for r in requests])
            action = action.reshape(self.action_spaces.shape)
            obs, reward, done, info = self.env.step(action)
            agent_data = [[obs, reward, done, None]]*len(self.sub_envs)
            info = info if len(info) == len(dones) else [info]*len(dones)
            agent_data = list(zip(obs, rewards, dones, info))
            self.mailbox.append(agent_data)
            return obs, rewards, dones, info
        else:
            raise RuntimeError('Requests are inconsistent.')
        
    def __next__(self):
        data = self.handle_requests()
        if data is None:
            raise StopIteration
        else:
            return data
        
    def __iter__(self):
        return self
