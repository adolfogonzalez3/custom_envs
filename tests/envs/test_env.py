'''Module to test single agent envs.'''
import pytest

from custom_envs.envs import SINGLE_AGENT_ENVIRONMENTS


@pytest.mark.parametrize("environment_class", SINGLE_AGENT_ENVIRONMENTS)
def test_step(environment_class):
    '''Tests environment's step method.'''
    environ = environment_class()
    action = environ.action_space.sample()
    state, reward, terminal, info = environ.step(action)
    assert environ.observation_space.contains(state)
    assert isinstance(reward, float)
    assert isinstance(terminal, bool)
    assert isinstance(info, dict)


@pytest.mark.parametrize("environment_class", SINGLE_AGENT_ENVIRONMENTS)
def test_reset(environment_class):
    '''Tests environment's reset method.'''
    environ = environment_class()
    state = environ.reset()
    assert environ.observation_space.contains(state)
