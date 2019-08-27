"""
A module for logging utilities and classes.
"""
import time
import logging
from collections import defaultdict
from pathlib import Path

import gym
from gym.core import Wrapper
import pandas as pd

LOGGER = logging.getLogger(__name__)

class Monitor(Wrapper):
    """
        A monitor wrapper for Gym environments, it is used to know the episode
        reward, length, time and other data.
    """
    EXT = ".mon.csv"
    file_handler = None

    def __init__(self, env, file_path, info_keywords=(), chunk_size=1,
                 callbacks=None):
        """
        A monitor wrapper for Gym environments, it is used to know the episode
        reward, length, time and other data.

        :param env: (Gym environment) The environment
        :param file_path: (str) the path to save a log file, can be None
                               for no log
        :param allow_early_resets: (bool) allows the reset of the environment
                                          before it is done
        :param reset_keywords: (tuple) extra keywords for the reset call, if
                                       extra parameters are needed at reset
        :param callback: (callable) Called every step and fed the current
                                    state, reward, done, and info.
        """
        if callable(env):
            env = env()
        Wrapper.__init__(self, env=env)
        self.t_start = time.time()
        if file_path is not None:
            self.file_path = Path(file_path).resolve().with_suffix(Monitor.EXT)
        else:
            self.file_path = None
        self.chunk_size = chunk_size

        self.info_keywords = info_keywords
        self.last_info = {}
        self.rewards = None
        self.metric_history = defaultdict(list)
        self.current_episode = 0
        self.callbacks = [] if callbacks is None else callbacks
        self.data = []

    def save(self):
        """
        Saves the data acquired so far.
        """
        if self.file_path is not None:
            header = not self.file_path.is_file()
            mode = 'w' if header else 'a'
            dataframe = pd.DataFrame(self.data)
            dataframe = dataframe.reindex(sorted(dataframe.columns), axis=1)
            dataframe.to_csv(self.file_path, header=header, index=False,
                             mode=mode)
        self.data = []

    def reset(self, **kwargs):
        """
        Calls the Gym environment reset. Can only be called if the environment
        is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if
                       defined by reset_keywords
        :return: ([int] or [float]) the first observation of the environment
        """
        self.rewards = []
        self.current_episode += 1
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Step the environment with the given action

        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward,
                                                           done, information
        """
        observation, reward, done, info = self.env.step(action)
        for callback in self.callbacks:
            callback({
                'observation': observation, 'reward': reward,
                'done': done, 'info': info, 'episode': self.current_episode
            })
        self.rewards.append(reward)
        if done:
            ep_rew = sum(self.rewards)
            eplen = len(self.rewards)
            ep_info = {
                "r": round(ep_rew, 6), "l": eplen,
                "t": round(time.time() - self.t_start, 6),
                'current_reward': reward, 'episode': self.current_episode
            }
            self.last_info = info
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.data.append(ep_info)
            if len(self.data) >= self.chunk_size:
                self.save()
            info['episode'] = ep_info
            self.metric_history['rewards'].append(ep_rew)
            self.metric_history['lengths'].append(eplen)
            self.metric_history['times'].append(time.time() - self.t_start)
        return observation, reward, done, info

    def close(self):
        """
        Closes the environment
        """
        if self.data:
            self.save()
        super().close()

    def get_total_steps(self):
        """
        Returns the total number of timesteps

        :return: (int)
        """
        return self.total_steps

    def get_episode_rewards(self):
        """
        Returns the rewards of all the episodes

        :return: ([float])
        """
        return self.metric_history.get('rewards', [])

    def get_episode_lengths(self):
        """
        Returns the number of timesteps of all the episodes

        :return: ([int])
        """
        return self.metric_history.get('lengths', [])

    def get_episode_times(self):
        """
        Returns the runtime in seconds of all the episodes

        :return: ([float])
        """
        return self.metric_history.get('rewards', [])


def create_env(env_name, log_dir=None, num_of_envs=1, **kwarg):
    '''
    Create many environments at once with optional independent logging.

    :param env_name: (str) The name of an OpenAI environment to create.
    :param log_dir: (str or None) If str then log_dir is the path to save all
                                  log files to otherwise if None then don't
                                  log anything.
    :param num_of_envs: (int) The number of environments to create.
    '''
    envs = [gym.make(env_name, **kwarg) for _ in range(num_of_envs)]
    if log_dir is not None:
        log_dir = Path(log_dir)
        envs = [Monitor(env, str(log_dir / str(i)), chunk_size=10,
                        info_keywords=('objective', 'accuracy'))
                for i, env in enumerate(envs)]
    return envs
