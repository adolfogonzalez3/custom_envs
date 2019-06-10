
from multiprocessing import Process
from threading import Thread

import pytest

from custom_envs.multiagent.mailboxdict import MailboxDict

JOB_EXE = [Thread, Process]
NUMBER_OF_PROCESSORS = 2


def task_get(mailbox):
    task_id = mailbox.get()
    mailbox.append(task_id)


def task_sudden_stop(mailbox):
    try:
        task_id = mailbox.get()
        if task_id != NUMBER_OF_PROCESSORS - 1:
            mailbox.append(task_id)
    finally:
        mailbox.close()


def task_sudden_error(mailbox):
    try:
        task_id = mailbox.get()
        if task_id != NUMBER_OF_PROCESSORS - 1:
            raise RuntimeError
    finally:
        mailbox.close()


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
    mailbox = MailboxDict()
    tasks = [job_exe(target=task_get, args=(mailbox.spawn(),),
                     daemon=True)
             for _ in range(NUMBER_OF_PROCESSORS)]
    for task in tasks:
        task.start()
    message = {i: i for i in range(NUMBER_OF_PROCESSORS)}
    mailbox.append(message)
    data = mailbox.get()
    assert(data == message)
    for task in tasks:
        task.join()


@pytest.mark.parametrize("job_exe", JOB_EXE)
def test_mailboxinsync_sudden_closing(job_exe):
    mailbox = MailboxDict()
    tasks = [job_exe(target=task_sudden_stop, args=(mailbox.spawn(),),
                     daemon=True)
             for _ in range(NUMBER_OF_PROCESSORS)]
    for task in tasks:
        task.start()
    message = {i: i for i in range(NUMBER_OF_PROCESSORS)}
    mailbox.append(message)
    if not mailbox.is_broken() and mailbox.poll():
        assert True
    for task in tasks:
        task.join()


@pytest.mark.parametrize("job_exe", JOB_EXE)
def test_mailboxinsync_sudden_error(job_exe):
    mailbox = MailboxDict()
    tasks = [job_exe(target=task_sudden_error, args=(mailbox.spawn(),),
                     daemon=True)
             for _ in range(NUMBER_OF_PROCESSORS)]
    for task in tasks:
        task.start()
    message = {i: i for i in range(NUMBER_OF_PROCESSORS)}
    mailbox.append(message)
    if not mailbox.is_broken() and mailbox.poll():
        assert True
    for task in tasks:
        task.join()


@pytest.mark.parametrize("job_exe", JOB_EXE)
def test_mailbox_recursion(job_exe):
    mailbox = MailboxDict()
    tasks = [job_exe(target=task_recursion, args=(mailbox.spawn(), 5))
             for _ in range(NUMBER_OF_PROCESSORS)]
    for task in tasks:
        task.start()
    assert mailbox.poll()
    assert all(mailbox.get().values())
    message = {i: i for i in range(NUMBER_OF_PROCESSORS)}
    mailbox.append(message)
    for task in tasks:
        task.join()

if __name__ == '__main__':
    test_mailboxinsync_data_correctness(Thread)
