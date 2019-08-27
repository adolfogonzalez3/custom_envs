'''Top level for problem submodule.'''

from custom_envs.problems.base_problem import BaseProblem
from custom_envs.problems.optimize_nn import OptimizeNN
from custom_envs.problems.optimize_function import OptimizeFunction

def get_problem(name='func', **kwargs):
    '''
    Return a problem.
    '''
    if name == 'nn':
        return OptimizeNN.create(**kwargs)
    elif name == 'func':
        return OptimizeFunction.create(**kwargs)
    else:
        raise RuntimeError('Not a name of a problem.')
