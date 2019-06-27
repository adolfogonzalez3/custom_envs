'''A module that contains a class for handling multiagent environments.'''
from threading import Thread
from collections import namedtuple
from enum import Enum

from gym import Env
from gym.spaces import Box
import numpy as np

from custom_envs.multiagent import MailboxDict

StepReturnType = namedtuple('StepReturnType',
                            ['states', 'rewards', 'terminals', 'infos'])
ResetReturnType = namedtuple('ResetReturnType', ['states'])
RenderReturnType = namedtuple('RenderReturnType', [])
CloseReturnType = namedtuple('CloseReturnType', [])


class RequestType(Enum):
    '''Contains all types a request may be.'''
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
        subsize = size // segments
        size_leftover = int(np.ceil(size / segments))
        low = np.min(space.low)
        high = np.max(space.high)
        leftover = [Box(low, high, shape=(size_leftover,), dtype=space.dtype)]
        return [Box(low, high, shape=(subsize,), dtype=space.dtype)
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

    def render(self, mode='human'):
        request = EnvRequest(RequestType.RENDER, None)
        self._mailbox.append(request)

    def close(self):
        request = EnvRequest(RequestType.CLOSE, None)
        self._mailbox.append(request)

    def __call__(self):
        return self


def run_handle(environment):
    '''Run handle until all experiments end.'''
    data = 0
    while data is not None:
        data = environment.handle_requests()


class MultiEnvServer:
    '''A class convert a Multi agent envs into many single agent envs.'''

    def __init__(self, env):
        self.main_environment = env
        self.mailbox = MailboxDict()
        self.sub_environments = {
            name: EnvSpawn(env.observation_space[name],
                           env.action_space[name],
                           self.mailbox.spawn(name))
            for name in env.action_space.spaces.keys()
        }

    def handle_requests(self, timeout=None):
        '''
        Handle any incoming messages.

        :param timeout: (int or None) If int then will poll for at least
                                      `timeout` seconds before erroring else
                                      if None then will wait indefinitely.
        :return: (dict) If a dictionary then will contain the current
                                data that has been received
        '''
        requests = self.mailbox.get(timeout=timeout)
        if requests is None:
            data = {}
        elif all([r.type == RequestType.RESET for r in requests.values()]):
            observations = self.main_environment.reset()
            self.mailbox.append(observations)
            data = self.handle_requests()
        elif all([r.type == RequestType.CLOSE for r in requests.values()]):
            data = None
        elif all([r.type == RequestType.STEP for r in requests.values()]):
            action = {name: rqst.data for name, rqst in requests.items()}
            states, rewards, dones, infos = self.main_environment.step(action)
            data = {name: (states[name], rewards[name], dones[name],
                           infos[name])
                    for name in states.keys()}
            self.mailbox.append(data)
        else:
            raise RuntimeError('Requests are inconsistent.')
        return data

    def __next__(self):
        data = self.handle_requests()
        if data:
            raise StopIteration
        return data

    def __iter__(self):
        return self

    def close(self):
        '''Close the environment.'''
        self.mailbox.close()
        self.main_environment.close()


class MultiEnvWrapper:
    '''A class convert a Multi agent envs into many single agent envs.'''

    def __init__(self, environment):
        self.observation_spaces = environment.observation_spaces
        self.action_spaces = environment.action_spaces
        server = MultiEnvServer(environment)
        self.sub_environments = server.sub_environments
        self._backend = Thread(target=run_handle, args=[server],
                               daemon=True)
        self._backend.start()

    def join(self, timeout=None):
        self._backend.join(timeout=timeout)
