
from enum import Enum

from collections import deque, namedtuple
from multiprocessing import Queue, Pipe
from multiprocessing.connection import wait
from time import sleep

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

def create_mailbox():
    host, client = Pipe(True)
    return Mailbox(host), Mailbox(client)


class MailboxInSync(object):
    '''A mailbox which acts as a consumer for multiple produceers.
    
    The mailbox will be able to generate MailboxSpawn objects which send data
    to _main_mailbox. Each instance of MailboxSpawn spawned from an instance
    of MailboxInSync also will consume data tagged with their ID.
    '''
    
    def __init__(self):
        self.mailboxes = []
        
    def spawn(self):
        owner, client = create_mailbox()
        self.mailboxes.append(owner)
        return client

    def is_broken(self):
        for mbox in self.mailboxes:
            if mbox.peek() is MailboxMsg.CLOSE:
                return True
        #broken_pipe_list = [mbox.peek() is MailboxMsg.CLOSE
        #                    for mbox in self.mailboxes]
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
        else:
            return None

    def close(self):
        self.append([True]*len(self.mailboxes))
        for mailbox in self.mailboxes:
            mailbox.close()

    def __enter__(self):
        pass

    def __exit__(self, a, b, c):
        pass
        
if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor

    def task(L):
        ID = L.get()
        print("ID: ", ID)
        L.append(ID)
        ID = L.get()
        print("Next ID: ", ID)
        L.append(ID)
        
    mailbox = MailboxInSync()
    with ThreadPoolExecutor() as executor:
        executor.submit(task, mailbox.spawn())
        executor.submit(task, mailbox.spawn())
        executor.submit(task, mailbox.spawn())
        executor.submit(task, mailbox.spawn())
        executor.submit(task, mailbox.spawn())
        
        mailbox.append([5-i for i in range(5)])
        
        data = mailbox.get()
        print(data)
        
        mailbox.append([i*2 for i in range(5)])
        
        data = mailbox.get()
        print(data)
