'''
Init file for custom_envs.envs subpackage.
'''
from custom_envs.envs.baseenvironment import BaseEnvironment
from custom_envs.envs.baseenvironment import BaseMultiEnvironment
from custom_envs.envs.multioptimize import MultiOptimize
from custom_envs.envs.multioptlrs import MultiOptLRs

SINGLE_AGENT_ENVIRONMENTS = (MultiOptimize, MultiOptLRs)
