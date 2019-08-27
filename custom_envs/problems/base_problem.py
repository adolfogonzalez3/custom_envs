'''A module that contains the BaseProblem abstract class.'''
from abc import abstractmethod
from collections import namedtuple

ProblemTuple = namedtuple('ProblemTuple', ['gradient', 'loss', 'parameters'])


class BaseProblem:
    '''
    An abstraction on problems which are used in the optimize environment.
    '''

    @classmethod
    def create(cls, *args, **kwargs):
        '''
        Create an instance of the class.

        Used in order to wrap the instances in convenience containers.
        '''
        return cls(*args, **kwargs)

    @abstractmethod
    def reset(self):
        '''
        Reset the Problem to its initial point.
        '''

    @abstractmethod
    def get_gradient(self):
        '''
        Retrieve the current gradient for the problem.
        '''

    @abstractmethod
    def get_loss(self):
        '''
        Retrieve the current loss for the problem.
        '''

    @abstractmethod
    def get_parameters(self):
        '''
        Retrieve the current parameters for the problem.
        '''

    @abstractmethod
    def set_parameters(self, parameters):
        '''
        Retrieve the current parameters for the problem.
        '''

    def next(self):
        '''
        Call for problems which require additional steps.
        '''

    def get(self):
        '''
        Retrieve the gradient, loss, and parameters for the problem.
        '''
        return ProblemTuple(
            self.get_gradient(), self.get_loss(), self.get_parameters()
        )

    @property
    def size(self):
        '''Return the number of parameters.'''
        return len(self.get_parameters())

    @property
    def parameters(self):
        '''Return the parameters of the problem.'''
        return self.get_parameters()
