'''
Init file for custom_envs package.

Contains all registering calls to gym.
'''
from gym.envs.registration import register

from custom_envs.data import load_data

# Experiments

register(
    id='Optimize-v0',
    entry_point='custom_envs.envs.optimize:Optimize'
)

register(
    id='OptLR-v0',
    entry_point='custom_envs.envs.optlr:OptLR'
)

register(
    id='OptLRs-v0',
    entry_point='custom_envs.envs.optlrs:OptLRs'
)

register(
    id='OptimizeCorrect-v0',
    entry_point='custom_envs.envs.optimizecorrect:OptimizeCorrect'
)

register(
    id='MultiOptLRs-v0',
    entry_point='custom_envs.envs.multioptlrs:MultiOptLRs'
)
