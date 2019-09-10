'''Module to test single agent envs.'''
import pytest

from custom_envs.envs import SINGLE_AGENT_ENVIRONMENTS


@pytest.mark.parametrize("environment_class", SINGLE_AGENT_ENVIRONMENTS)
def test_step(environment_class):
    '''Tests environment's step method.'''
    environ = environment_class()
    environ.reset()
    assert environ.current_step == 0
    action = environ.action_space.sample()
    terminal = False
    for i in range(1, 10):
        state, reward, terminal, info = environ.step(action)
        assert environ.current_step == i
        assert environ.observation_space.contains(state)
        assert isinstance(reward, float)
        assert isinstance(terminal, bool)
        assert isinstance(info, dict)
        if terminal:
            break


@pytest.mark.parametrize("environment_class", SINGLE_AGENT_ENVIRONMENTS)
def test_reset(environment_class):
    '''Tests environment's reset method.'''
    environ = environment_class()
    state = environ.reset()
    assert environ.observation_space.contains(state)
