'''Module which tests OptimizerNN.'''
import numpy as np

from custom_envs.problems import OptimizeNN


def test_reset():
    '''Test the reset method.'''
    problem = OptimizeNN.create()
    old_params = problem.get_parameters()
    problem.reset()
    assert problem.data_set_iter
    assert problem.current_batch
    params = problem.get_parameters()
    assert not np.all(old_params == params)
