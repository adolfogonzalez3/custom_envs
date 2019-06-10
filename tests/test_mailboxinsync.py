
from multiprocessing import Process
from threading import Thread

import pytest

from custom_envs.multiagent.MailboxInSync import MailboxInSync

JOB_EXE = [Thread, Process]
NUMBER_OF_PROCESSORS = 2


def task(L):
    with L:
        task_id = L.get()
        L.append(task_id)


def task_sudden_stop(L):
    with L:
        task_id = L.get()
        if task_id != NUMBER_OF_PROCESSORS - 1:
            L.append(task_id)


def task_sudden_error(L):
    with L:
        task_id = L.get()
        if task_id != NUMBER_OF_PROCESSORS - 1:
            raise RuntimeError


def task_recursion(mailbox, level):
    if level == 0:
        mailbox.append(True)
        mailbox.get()
    else:
        proc = Process(target=task_recursion, args=(mailbox, level-1))
        proc.start()
        proc.join()


@pytest.mark.parametrize("job_exe", JOB_EXE)
def test_mailboxinsync_data_correctness(job_exe):
    mailbox = MailboxInSync()
    tasks = [job_exe(target=task, args=(mailbox.spawn(),),
                     daemon=True)
             for _ in range(NUMBER_OF_PROCESSORS)]
    for t in tasks:
        t.start()
    mailbox.append([i for i in range(NUMBER_OF_PROCESSORS)])
    data = sorted(mailbox.get())
    assert(data == [i for i in range(NUMBER_OF_PROCESSORS)])
    for t in tasks:
        t.join()


@pytest.mark.parametrize("job_exe", JOB_EXE)
def test_mailboxinsync_sudden_closing(job_exe):
    mailbox = MailboxInSync()
    tasks = [job_exe(target=task_sudden_stop, args=(mailbox.spawn(),),
                     daemon=True)
             for _ in range(NUMBER_OF_PROCESSORS)]
    for t in tasks:
        t.start()
    mailbox.append([i for i in range(NUMBER_OF_PROCESSORS)])
    if not mailbox.is_broken() and mailbox.poll():
        assert True
    for t in tasks:
        t.join()


@pytest.mark.parametrize("job_exe", JOB_EXE)
def test_mailboxinsync_sudden_error(job_exe):
    mailbox = MailboxInSync()
    tasks = [job_exe(target=task_sudden_error, args=(mailbox.spawn(),),
                     daemon=True)
             for _ in range(NUMBER_OF_PROCESSORS)]
    for t in tasks:
        t.start()
    mailbox.append([i for i in range(NUMBER_OF_PROCESSORS)])
    if not mailbox.is_broken() and mailbox.poll():
        assert True
    for t in tasks:
        t.join()


@pytest.mark.parametrize("job_exe", JOB_EXE)
def test_mailbox_recursion(job_exe):
    mailbox = MailboxInSync()
    tasks = [job_exe(target=task_recursion, args=(mailbox.spawn(), 5))
             for _ in range(NUMBER_OF_PROCESSORS)]
    for t in tasks:
        t.start()
    assert mailbox.poll()
    assert all(mailbox.get())
    mailbox.append([i for i in range(NUMBER_OF_PROCESSORS)])
    for t in tasks:
        t.join()
