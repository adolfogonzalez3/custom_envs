'''Module for MailboxDict implementation.'''
import multiprocessing as mp
from enum import Enum
from itertools import chain

from custom_envs.networking import PipeQueue


class MailboxMsg(Enum):
    ERROR = 0
    EMPTY = 1
    CLOSE = 2


class MailboxRemote:
    '''A class for creating a remote mailbox.'''

    def __init__(self, pipe):
        self.pipe = pipe
        self.data = MailboxMsg.EMPTY

    def append(self, data):
        '''
        Send data to the main mailbox.
        '''
        self.pipe.send(data)

    def peek(self, timeout=None):
        '''
        Poll the mailbox to check for an incoming message.

        :param timeout: (int or None) If int then will poll for at least
                                      `timeout` seconds before erroring else
                                      if None then will wait indefinitely.
        '''
        if self.data is MailboxMsg.EMPTY:
            if self.poll(timeout=timeout):
                data = self.pipe.recv()
                self.data = data
            else:
                raise EOFError('No objects in Pipe.')
        return self.data

    def poll(self, timeout=None):
        '''
        Poll the mailboxes to check for any incoming messages.

        :param timeout: (int or None) If int then will poll for at least
                                      `timeout` seconds before erroring else
                                      if None then will wait indefinitely.
        '''
        return self.pipe.poll(timeout=timeout)

    def get(self, timeout=None):
        '''
        Get any messages.

        :param timeout: (int or None) If int then will poll for at least
                                      `timeout` seconds before erroring else
                                      if None then will wait indefinitely.
        '''
        data = self.peek(timeout=timeout)
        self.data = MailboxMsg.EMPTY
        return data

    def close(self):
        self.pipe.send(MailboxMsg.CLOSE)


def create_mailbox(manager):
    #host, client = create_pipe()
    host = manager.Queue()
    client = manager.Queue()
    pipe = PipeQueue(host, client)
    return MailboxRemote(pipe), MailboxRemote(pipe.reverse())
    # return Mailbox(host), Mailbox(client)


class MailboxDict(object):
    '''A mailbox which acts as a consumer for multiple produceers.

    The mailbox will be able to generate MailboxSpawn objects which send data
    to _main_mailbox. Each instance of MailboxSpawn spawned from an instance
    of MailboxInSync also will consume data tagged with their ID.
    '''

    def __init__(self):
        self.mailboxes = {}
        self.manager = mp.Manager()

    def spawn(self, name=None):
        '''
        Spawn a new mail box.

        :param name: (str or None) The name of the new mailbox. If str then
                                   the mailbox is named name. Else if None
                                   then is named a number that is one greater
                                   than the largest number in the collection
                                   of mailboxes.
        '''
        name = max(i for i in chain([-1], self.mailboxes.keys())
                   if isinstance(i, int)) + 1 if name is None else name
        owner, client = create_mailbox(self.manager)
        self.mailboxes[name] = owner
        return client

    def is_broken(self):
        '''
        Check if any of the mailboxes have broken connections.
        '''
        return any(mbox.peek() is MailboxMsg.CLOSE
                   for mbox in self.mailboxes.values())

    def poll(self, timeout=None):
        '''
        Poll the mailboxes to check for any incoming messages.

        :param timeout: (int or None) If int then will poll for at least
                                      `timeout` seconds before erroring else
                                      if None then will wait indefinitely.
        '''
        return all([mbox.poll(timeout) for mbox in self.mailboxes.values()])

    def append(self, data):
        '''
        Send data to the mailboxes.

        :param data: (dict) A dictionary containing the data to send to the
                            other mailboxes.
        '''
        for name, value in data.items():
            self.mailboxes[name].append(value)

    def _unsafe_get(self):
        '''
        Get data from the mailboxes without checking if connections are valid.
        '''
        return {name: mbox.get() for name, mbox in self.mailboxes.items()}

    def get(self, timeout=None):
        '''
        Get data from the mailboxes.

        :param timeout: (int or None) If int then will poll for at least
                                      `timeout` seconds before erroring else
                                      if None then will wait indefinitely.
        '''
        is_ready = all(mbox.poll(timeout) for mbox in self.mailboxes.values())

        if is_ready:
            return self._unsafe_get()
        else:
            RuntimeError('Connections timed out')

    def close(self):
        '''
        Close all connections.
        '''
        self.append({name: True for name in self.mailboxes})
        for mailbox in self.mailboxes.values():
            mailbox.close()
