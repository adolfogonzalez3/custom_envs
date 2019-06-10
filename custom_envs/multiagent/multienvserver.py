
from copy import deepcopy
from collections import namedtuple
from enum import Enum

from gym import Env
from gym.spaces import Box
import numpy as np

from custom_envs.multiagent.MailboxInSync import MailboxInSync

StepReturnType = namedtuple('StepReturnType',
                            ['states', 'rewards', 'terminals', 'infos'])
ResetReturnType = namedtuple('ResetReturnType', ['states'])
RenderReturnType = namedtuple('RenderReturnType', [])
CloseReturnType = namedtuple('CloseReturnType', [])


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
        size = np.prod(space.shape)
        n = size // segments
        n_leftover = int(np.ceil(size / segments))
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


class MultiEnvServer:
    '''A class convert a Multi agent envs into many single agent envs.'''

    def __init__(self, environment):
        self.main_environment = environment
        self.mailbox = MailboxInSync()
        self.observation_spaces = environment.observation_spaces
        self.action_spaces = environment.action_spaces
        self.sub_environments = {name:
                                 EnvSpawn(self.observation_spaces[name],
                                          self.action_spaces[name],
                                          self.mailbox.spawn())
                                 for name in self.action_spaces.keys()
                                 }

    def handle_requests(self, timeout=None):
        requests = self.mailbox.get(timeout=timeout)
        if requests is None:
            data = []
        elif all([r.type == RequestType.RESET for r in requests]):
            obs = self.main_environment.reset()
            self.mailbox.append([obs]*len(self.sub_environments))
            data = self.handle_requests()
        elif all([r.type == RequestType.CLOSE for r in requests]):
            data = None
        elif all([r.type == RequestType.STEP for r in requests]):
            action = np.concatenate([r.data for r in requests])
            action = action.reshape(self.main_environment.action_space.shape)
            data = self.main_environment.step(action)
            agent_data = [deepcopy(data)
                          for _ in enumerate(self.sub_environments)]
            self.mailbox.append(agent_data)
        else:
            raise RuntimeError('Requests are inconsistent.')
        return data

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
        self.main_environment.close()

    def __enter__(self):
        pass
