
from multiprocessing import Process
from threading import Thread

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import pytest



from custom_envs.multiagent.MailboxInSync import MailboxInSync

EXECUTORS = (ThreadPoolExecutor, ProcessPoolExecutor)
JOB_EXE = [Process, Thread]
NUMBER_OF_PROCESSORS = 2

def task(L):
    with L:
        ID = L.get()
        L.append(ID)

def task_sudden_stop(L):
    with L:
        ID = L.get()
        if ID != NUMBER_OF_PROCESSORS - 1:
            L.append(ID)

def task_sudden_error(L):
    with L:
        ID = L.get()
        if ID != NUMBER_OF_PROCESSORS - 1:
            raise RuntimeError

@pytest.mark.parametrize("pool_executor", EXECUTORS)
def test_mailboxinsync_data_correctness_0(pool_executor):
    mailbox = MailboxInSync()
    with pool_executor(2) as executor:
        executor.map(task, [mailbox.spawn() for _ in range(2)])
        mailbox.append([i for i in range(NUMBER_OF_PROCESSORS)])

        data = sorted(mailbox.get())
        assert(sorted(data) == [i for i in range(2)])


@pytest.mark.parametrize("pool_executor", EXECUTORS)
def test_mailboxinsync_sudden_closing(pool_executor):
    mailbox = MailboxInSync()
    with pool_executor(2) as executor:
        executor.map(task_sudden_stop, [mailbox.spawn() for _ in range(2)])
        mailbox.append([i for i in range(NUMBER_OF_PROCESSORS)])

        if not mailbox.is_broken() and mailbox.poll():
            assert True

@pytest.mark.parametrize("pool_executor", EXECUTORS)
def test_mailboxinsync_sudden_error(pool_executor):
    mailbox = MailboxInSync()
    with pool_executor(2) as executor:
        executor.map(task_sudden_error, [mailbox.spawn() for _ in range(2)])
        mailbox.append([i for i in range(NUMBER_OF_PROCESSORS)])

        if not mailbox.is_broken() and mailbox.poll():
            assert True

if __name__ == '__main__':
    test_mailboxinsync_sudden_error(ProcessPoolExecutor)