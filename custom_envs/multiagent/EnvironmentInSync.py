
from copy import deepcopy
from collections import namedtuple
from enum import Enum

from gym import Env
from gym.spaces import Box
import numpy as np

from custom_envs.multiagent.MailboxInSync import MailboxInSync

class RequestType(Enum):
    STEP = 0
    RESET = 1
    RENDER = 2
    CLOSE = 3


def segment_space(space, segments):
    '''
    Divide a space into several segments.

    :param space: (A subclass of gym.Space) The space to segment
    :param title: (int) The number of segments.
    '''
    if isinstance(space, Box):
        n = np.prod(space.shape) // segments
        n_leftover = int(np.ceil(np.prod(space.shape) / segments))
        low = np.min(space.low)
        high = np.max(space.high)
        leftover = [Box(low, high, shape=(n_leftover,), dtype=space.dtype)]
        return [Box(low, high, shape=(n,), dtype=space.dtype)
                for i in range(segments-1)] + leftover

EnvRequest = namedtuple('EnvRequest', ['type', 'data'])

class EnvSpawn(Env):
    '''
    A class that is used to send and receive messages from the main Env.
    '''
    def __init__(self, observation_space, action_space, mailbox):
        self._mailbox = mailbox
        self.action_space = action_space
        self.observation_space = observation_space

    def start_connection(self):
        self._mailbox.append(EnvRequest(RequestType.START, None))
        
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

    def __call__(self):
        return self


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
        
    def handle_requests(self, timeout=None):
        requests = self.mailbox.get(timeout=timeout)
        if requests is None:
            return []
        elif all([r.type == RequestType.RESET for r in requests]):
            obs = self.env.reset()
            self.mailbox.append([obs]*len(self.sub_envs))
            return self.handle_requests()
        elif all([r.type == RequestType.CLOSE for r in requests]):
            return None
        elif all([r.type == RequestType.STEP for r in requests]):
            action = np.concatenate([r.data for r in requests])
            action = action.reshape(self.env.action_space.shape)
            data = self.env.step(action)
            agent_data = [deepcopy(data) for _ in enumerate(self.sub_envs)]
            self.mailbox.append(agent_data)
            return data
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

    def close(self):
        self.mailbox.close()
        self.env.close()

    def __enter__(self):
        pass
