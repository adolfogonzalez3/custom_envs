
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
    #host, client = create_pipe()
    host = manager.Queue()
    client = manager.Queue()
    pipe = PipeQueue(host, client)
    return Mailbox(pipe), Mailbox(pipe.reverse())
    #return Mailbox(host), Mailbox(client)


class MailboxInSync(object):
    '''A mailbox which acts as a consumer for multiple produceers.

    The mailbox will be able to generate MailboxSpawn objects which send data
    to _main_mailbox. Each instance of MailboxSpawn spawned from an instance
    of MailboxInSync also will consume data tagged with their ID.
    '''

    def __init__(self):
        self.mailboxes = []
        self.manager = mp.Manager()
        #self.manager.start()
        self.manager.get_server().serve_forever()

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


NUM_STEPS = 10**2

def task(L):
    for i in range(NUM_STEPS):
        L.append(i)
        L.get()

def __main():
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    from multiprocessing import Process
    from threading import Thread
    from time import time

    mailbox = MailboxInSync()
    begin = time()
    tasks = [Process(target=task, args=(mailbox.spawn(),))
             for _ in range(3)]
    for t in tasks:
        t.start()
    print('Loaded in {:.6} seconds.'.format(time() - begin))
    begin = time()
    for _ in range(NUM_STEPS):
        _ = mailbox.get()
        mailbox.append([i for i in range(3)])
    for t in tasks:
        t.join()
    time_elapsed = time() - begin
    time_per_transfer = time_elapsed / NUM_STEPS
    print(('Done in {:.6} seconds or {:.6} seconds per '
           'transfer.').format(time_elapsed, time_per_transfer))


if __name__ == '__main__':
    __main()
