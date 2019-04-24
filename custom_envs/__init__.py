
from custom_envs.data import load_data

from gym.envs.registration import register

# Experiments

register(
    id='Optimize-v0',
    entry_point='custom_envs.optimize:Optimize'
)

register(
    id='OptLR-v0',
    entry_point='custom_envs.optlr:OptLR'
)

register(
    id='OptLRs-v0',
    entry_point='custom_envs.optlrs:OptLRs'
)