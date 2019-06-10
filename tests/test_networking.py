'''Tests utils_networking module.'''

from time import sleep
from multiprocessing import Process

import pytest
import numpy as np
import numpy.random as npr

import custom_envs.networking as networking


def task_send(pipe):
    counter = 0
    for _ in range(10):
        counter += pipe.recv()
    pipe.send(counter)


def task_recv(pipe):
    for i in range(10):
        pipe.send(i)


def task_poll(pipe):
    sleep(3)
    pipe.send(True)

def task_recursion(pipe, level):
    if level == 0:
        pipe.send(True)
        pipe.recv()
    else:
        proc = Process(target=task_recursion, args=(pipe, level-1))
        proc.start()
        proc.join()

def test_pipe_send():
    host, client = networking.create_pipe()
    proc = Process(target=task_send, args=(client,))
    proc.start()
    for i in range(10):
        host.send(i)
    assert host.recv() == sum(range(10))
    proc.join()


def test_pipe_recv():
    host, client = networking.create_pipe()
    proc = Process(target=task_recv, args=(client,))
    proc.start()
    counter = 0
    for _ in range(10):
        counter += host.recv()
    assert counter == sum(range(10))
    proc.join()


def test_pipe_poll():
    host, client = networking.create_pipe()
    proc = Process(target=task_poll, args=(client,))
    proc.start()
    assert host.poll()
    assert host.recv()
    proc.join()


def test_pipe_recursion():
    host, client = networking.create_pipe()
    proc = Process(target=task_recursion, args=(client, 3))
    proc.start()
    assert host.poll()
    assert host.recv()
    host.send(3)
    proc.join()
