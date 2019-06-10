
import pytest

from custom_envs.envs import MULTI_AGENT_ENVIRONMENTS


@pytest.mark.parametrize("environment_class", MULTI_AGENT_ENVIRONMENTS)
def test_step(environment_class):
    environ = environment_class()
    action = {name: i for i, name in enumerate(environ.action_spaces.keys())}
    states, rewards, terminals, infos = environ.step(action)
    assert states.keys() == environ.action_spaces.keys()
    assert rewards.keys() == environ.action_spaces.keys()
    assert all(isinstance(reward, float) for reward in rewards.values())
    assert terminals.keys() == environ.action_spaces.keys()
    assert all(isinstance(terminal, bool) for terminal in terminals.values())
    assert infos.keys() == environ.action_spaces.keys()
    assert all(isinstance(info, dict) for info in infos.values())

@pytest.mark.parametrize("environment_class", MULTI_AGENT_ENVIRONMENTS)
def test_reset(environment_class):
    environ = environment_class()
    states = environ.reset()
    assert states.keys() == environ.action_spaces.keys()
