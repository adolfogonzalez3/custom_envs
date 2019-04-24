
from enum import Enum
from abc import ABC, abstractmethod

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
        else:
            data = self.data
            self.data = PipeMsg.EMPTY
            return data

    def poll(self, timeout=None):
        if self.data is PipeMsg.EMPTY:
            self.data = self.host.get(timeout=timeout)
        return True
        #return not self.host.empty()

    def reverse(self):
        return PipeQueue(self.client, self.host)
