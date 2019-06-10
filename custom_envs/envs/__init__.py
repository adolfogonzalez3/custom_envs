'''
Init file for custom_envs.envs subpackage.
'''
from custom_envs.envs.baseenvironment import BaseEnvironment
from custom_envs.envs.baseenvironment import BaseMultiEnvironment
from custom_envs.envs.optimize import Optimize
from custom_envs.envs.optimizecorrect import OptimizeCorrect
from custom_envs.envs.multioptlrs import MultiOptLRs

SINGLE_AGENT_ENVIRONMENTS = (Optimize, OptimizeCorrect)
MULTI_AGENT_ENVIRONMENTS = (MultiOptLRs,)
