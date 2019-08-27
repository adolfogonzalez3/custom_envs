'''Module which tests BaseProblem's descendants.'''
import pytest
import numpy as np
import numpy.random as npr

from custom_envs.problems import BaseProblem


def get_descendants(class_obj):
    '''
    Gather all descendants of a class.

    :param class_obj: The ancestor of the gathered list.
    :return: A list of descendant classes.
    '''
    classes = class_obj.__subclasses__()
    descendants = []
    while classes:
        class_obj = classes.pop()
        descendants.append(class_obj)
        classes.extend(class_obj.__subclasses__())
    return descendants


def compare_lists(list_1, list_2):
    '''Used to compare lists.'''
    list_cmp = [
        np.isclose(l1, l2, atol=1e-6) for l1, l2 in zip(list_1, list_2)
    ]
    indices = [i for i, truth in enumerate(list_cmp) if not truth]
    return not bool(indices)


DESCENDANTS = get_descendants(BaseProblem)


@pytest.mark.parametrize("problem_cls", DESCENDANTS)
def test_reset(problem_cls):
    '''Test the reset method.'''
    problem = problem_cls.create()
    old_params = npr.rand(problem.size)
    problem.set_parameters(old_params)
    problem.reset()
    new_params = problem.get_parameters()
    assert not all([g == o for g, o in zip(old_params, new_params)])


@pytest.mark.parametrize("problem_cls", DESCENDANTS)
def test_get_gradient(problem_cls):
    '''Test the get_gradient method.'''
    problem = problem_cls.create()
    gradient = problem.get_gradient()
    assert np.array(gradient).ndim == 1


@pytest.mark.parametrize("problem_cls", DESCENDANTS)
def test_get_loss(problem_cls):
    '''Test the get_loss method.'''
    problem = problem_cls.create()
    loss = problem.get_loss()
    assert float(loss)


@pytest.mark.parametrize("problem_cls", DESCENDANTS)
def test_get_parameters(problem_cls):
    '''Test the get_parameters method.'''
    problem = problem_cls.create()
    parameters = problem.get_parameters()
    assert np.array(parameters).ndim == 1


@pytest.mark.parametrize("problem_cls", DESCENDANTS)
def test_set_parameters(problem_cls):
    '''Test the set_parameters method.'''
    problem = problem_cls.create()
    other_parameters = npr.rand(problem.size)
    problem.set_parameters(other_parameters)
    parameters = problem.get_parameters()
    assert compare_lists(parameters, other_parameters)


@pytest.mark.parametrize("problem_cls", DESCENDANTS)
def test_get(problem_cls):
    '''Test the get method.'''
    problem = problem_cls.create()
    other_gradient = problem.get_gradient()
    other_loss = problem.get_loss()
    other_parameters = problem.get_parameters()
    gradient, loss, parameters = problem.get()
    assert compare_lists(gradient, other_gradient)
    assert np.isclose(loss, other_loss)
    assert compare_lists(parameters, other_parameters)
