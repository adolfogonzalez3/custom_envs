
import multiprocessing as mp
from enum import Enum
from collections import namedtuple

from custom_envs.networking import create_pipe, PipeQueue

SpawnData = namedtuple('SpawnData', ['ID', 'data'])
SpawnMailbox = namedtuple('SpawnMailbox', ['ID', 'mailbox'])


class MailboxMsg(Enum):
    ERROR = 0
    EMPTY = 1
    CLOSE = 2


class Mailbox:
    def __init__(self, pipe):
        self.pipe = pipe
        self.data = MailboxMsg.EMPTY

    def append(self, data):
        self.pipe.send(data)

    def peek(self, timeout=None):
        if self.data is MailboxMsg.EMPTY:
            if self.poll(timeout=timeout):
                data = self.pipe.recv()
                self.data = data
            else:
                raise EOFError('No objects in Pipe.')
        return self.data

    def poll(self, timeout=None):
        return self.pipe.poll(timeout=timeout)

    def get(self, timeout=None):
        data = self.peek(timeout=timeout)
        self.data = MailboxMsg.EMPTY
        return data

    def close(self):
        self.pipe.send(MailboxMsg.CLOSE)

    def __enter__(self):
        pass

    def __exit__(self, a, b, c):
        self.close()


def create_mailbox(manager):
    host = manager.Queue()
    client = manager.Queue()
    pipe = PipeQueue(host, client)
    return Mailbox(pipe), Mailbox(pipe.reverse())


class MailboxInSync(object):
    '''A mailbox which acts as a consumer for multiple produceers.

    The mailbox will be able to generate MailboxSpawn objects which send data
    to _main_mailbox. Each instance of MailboxSpawn spawned from an instance
    of MailboxInSync also will consume data tagged with their ID.
    '''

    def __init__(self):
        self.mailboxes = []
        self.manager = mp.Manager()

    def spawn(self):
        owner, client = create_mailbox(self.manager)
        self.mailboxes.append(owner)
        return client

    def is_broken(self):
        for mbox in self.mailboxes:
            if mbox.peek() is MailboxMsg.CLOSE:
                return True
        return True

    def poll(self, timeout=None):
        return all([mailbox.poll(timeout) for mailbox in self.mailboxes])

    def append(self, data, unequal=False):
        if unequal is False and len(data) != len(self.mailboxes):
            raise RuntimeError(("The length of data isn't equal to the number"
                                " of spawned instances."))
        for mailbox, data in zip(self.mailboxes, data):
            mailbox.append(data)

    def _unsafe_get(self):
        return [mailbox.get() for mailbox in self.mailboxes]

    def get(self, timeout=None):
        is_ready = True
        for mailbox in self.mailboxes:
            is_ready = is_ready and mailbox.poll(timeout)
        if is_ready:
            return self._unsafe_get()

    def close(self):
        self.append([True]*len(self.mailboxes))
        for mailbox in self.mailboxes:
            mailbox.close()

    def __enter__(self):
        pass

    def __exit__(self, a, b, c):
        pass
