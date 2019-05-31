
from enum import Enum
from abc import ABC, abstractmethod

import zmq

class BasePipe(ABC):
    @abstractmethod
    def send(self, data):
        pass

    @abstractmethod
    def recv(self):
        pass

    @abstractmethod
    def poll(self, timeout=None):
        pass

    @abstractmethod
    def close(self):
        pass

class PipeMsg(Enum):
    EMPTY = 0

class PipeQueue(BasePipe):
    def __init__(self, host, client):
        self.host = host
        self.client = client
        self.data = PipeMsg.EMPTY

    def send(self, data):
        self.client.put(data)

    def recv(self):
        if self.data is PipeMsg.EMPTY:
            return self.host.get()
        data = self.data
        self.data = PipeMsg.EMPTY
        return data

    def poll(self, timeout=None):
        if self.data is PipeMsg.EMPTY:
            self.data = self.host.get(timeout=timeout)
        return True

    def reverse(self):
        return PipeQueue(self.client, self.host)

    def close(self):
        pass


class ZMQQueueServer(BasePipe):
    def __init__(self, host_name='tcp://127.0.0.1'):
        ctx = zmq.Context.instance()
        self.socket = ctx.socket(zmq.PAIR)
        port = self.socket.bind_to_random_port(host_name)
        self.host_name = '{}:{:d}'.format(host_name, port)

    def send(self, data):
        self.socket.send_pyobj(data)

    def recv(self):
        return self.socket.recv_pyobj()

    def poll(self, timeout=None):
        return self.socket.poll(timeout) > 0

    def create_client(self):
        return ZMQQueueClient(self.host_name)

    def close(self):
        self.socket.close()


class ZMQQueueClient(BasePipe):
    def __init__(self, host_name):
        self.socket = None
        self.host_name = host_name

    def open(self):
        ctx = zmq.Context.instance()
        self.socket = ctx.socket(zmq.PAIR)
        self.socket.connect(self.host_name)

    def send(self, data):
        if self.socket is None:
            self.open()
        self.socket.send_pyobj(data)

    def recv(self):
        if self.socket is None:
            self.open()
        return self.socket.recv_pyobj()

    def poll(self, timeout=None):
        if self.socket is None:
            self.open()
        return self.socket.poll(timeout) > 0

    def close(self):
        if self.socket is not None:
            self.socket.close()

def create_pipe():
    host = ZMQQueueServer()
    return host, host.create_client()
