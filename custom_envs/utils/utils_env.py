'''A module for storing functions used by the Optimize Environments.'''

import numpy as np
from gym.spaces import Box

from custom_envs.utils.utils_common import History


def get_obs_version(shape, max_history, version=0):
    '''
    Get the Space and History object depending on the version.

    This function serves to centralize the different methods of creating a
    history object.

    :param shape: ((int,...)) A tuple of ints that represents the shape of
                              model.
    :param version: (int) The version of the History object to retrieve.
    :return: (gym.Space, History) Returns a space that inherits gym.Space and a
                                  History object.
    '''
    if version == 0:
        history = History(1, gradients=shape)
        space = Box(low=-1e6, high=1e6, dtype=np.float32, shape=(1,))
    elif version == 1:
        history = History(max_history, losses=(), gradients=shape)
        space = Box(low=-1e6, high=1e6, dtype=np.float32,
                    shape=(2*max_history,))
    elif version == 2:
        history = History(1, weights=shape, losses=(), gradients=shape)
        space = Box(low=-1e6, high=1e6, dtype=np.float32, shape=(3,))
    elif version == 3:
        history = History(max_history, weights=shape, losses=(),
                          gradients=shape)
        space = Box(low=-1e6, high=1e6, dtype=np.float32,
                    shape=(3*max_history,))
    elif version == 4:
        history = History(max_history, gradients=shape)
        space = Box(low=-1e6, high=1e6, dtype=np.float32, shape=(max_history,))
    elif version == 5:
        history = History(max_history, losses=(),
                          gradients=shape, actions=shape)
        space = Box(low=-1e6, high=1e6, dtype=np.float32,
                    shape=(3*max_history,))
    else:
        raise RuntimeError()
    return space, history


def get_action_space_optlrs(version=0):
    '''
    Get the bounds for the version of the OPTLRS family of environments.

    This function serves to centralize the different methods of choosing the
    action space for the OPTLRS family.

    :param version: (int) The version of the actions to retrieve.
    :return: (gym.Space) A space that inherits gym.Space.
    '''
    if version == 0:
        space = Box(low=-4., high=4., dtype=np.float32, shape=(1,))
    elif version == 1:
        space = Box(low=0., high=1e4, dtype=np.float32, shape=(1,))
    else:
        raise RuntimeError()
    return space


def get_reward(loss, adjusted_loss, version=0):
    '''
    Get the reward depending on the version.

    This function serves to centralize the different methods of choosing the
    reward.

    :param normal: (float) The loss of the model.
    :param adjusted_loss: (float) The loss of the model adjusted.
    :param version: (int) The version of the reward function to use.
    :return: (float) The reward.
    '''
    if version == 0:
        reward = -float(adjusted_loss)
    elif version == 1:
        reward = float(1 / loss)
    elif version == 2:
        reward = -float(adjusted_loss) * 100
    elif version == 3:
        reward = float(1 / loss) * 100
    elif version == 4:
        reward = np.log(1 / loss)
    else:
        raise RuntimeError()
    return reward


def get_action_optlrs(action, version):
    '''
    Get the action depending on the version.

    This function serves to centralize the different methods of choosing the
    action for the OPTLRS family.

    :param action: (numpy.array) The actions which the agent decided.
    :param version: (int) The version to use to compute the action.
    :return: (numpy.array) The action.
    '''
    if version == 0:
        action = 10**(action-4)
    elif version == 1:
        action = action*1e-3
    elif version == 2:
        action = 2**action
    else:
        raise RuntimeError()
    return action


def get_observation(history, version=0):
    '''
    Get the observation depending on the version.

    This function serves to centralize the different methods of computing the
    observation.

    :param history: (History) The history object for the model.
    :param version: (int) The version to use to compute the observation.
    :return: (numpy.array) The observation.
    '''
    past_losses = history['losses']
    past_grads = history['gradients']
    past_weights = history['weights']
    adjusted_loss = past_losses[0] / (np.abs(past_losses[1]) + 1e-3)
    adjusted_wght = past_weights[0] / (np.abs(past_weights[1]) + 1e-3)
    adjusted_grad = past_grads[0] / (np.abs(past_grads[1]) + 1e-3)
    if version == 0:
        adjusted_grad = past_grads[0] / (np.abs(past_grads[1]) + 1e-3)
    elif version == 1:
        adjusted_grad = past_grads[0] * 1e2
    elif version == 2:
        prev_loss = np.abs(past_losses[1] - past_losses[2])
        adjusted_loss = (past_losses[0] - past_losses[1]) / (prev_loss + 1e-3)
        prev_wght = np.abs(past_weights[1] - past_weights[2])
        abs_wght = np.abs(past_weights[0] - past_weights[1])
        adjusted_wght = prev_wght / (abs_wght + 1e-8)
        prev_grad = np.abs(past_grads[1] - past_grads[2])
        adjusted_grad = (past_grads[0] - past_grads[1]) / (prev_grad + 1e-3)
    elif version == 3:
        adjusted_grad = past_grads[0] / np.abs(past_grads[1])
        adjusted_grad = np.nan_to_num(adjusted_grad)
        adjusted_wght = past_weights[0] / np.abs(past_weights[1])
        adjusted_wght = np.nan_to_num(adjusted_wght)
        adjusted_loss = past_losses[0] / np.abs(past_losses[1])
        adjusted_loss = np.nan_to_num(adjusted_loss)
    else:
        raise RuntimeError()
    return float(adjusted_loss), adjusted_wght, adjusted_grad
