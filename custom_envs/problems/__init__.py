'''Top level for problem submodule.'''

from custom_envs.problems.base_problem import BaseProblem
from custom_envs.problems.optimize_nn import OptimizeNN

def get_problem(name='nn', **kwargs):
    '''
    Return a problem.
    '''
    if name == 'nn':
        return OptimizeNN.create(**kwargs)
    else:
        raise RuntimeError('Not a name of a problem.')
