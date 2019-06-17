"""
A module for logging utilities and classes.
"""
import time
from pathlib import Path

from gym.core import Wrapper
import pandas as pd


class Monitor(Wrapper):
    """
        A monitor wrapper for Gym environments, it is used to know the episode
        reward, length, time and other data.
    """
    EXT = ".mon.csv"
    file_handler = None

    def __init__(self, env, file_path, allow_early_resets=False,
                 reset_keywords=(), info_keywords=(), chunk_size=1):
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
        :param info_keywords: (tuple) extra information to log, from the
                                      information return of environment.step
        """
        Wrapper.__init__(self, env=env)
        self.t_start = time.time()
        if file_path is not None:
            self.file_path = Path(file_path).resolve().with_suffix(Monitor.EXT)
        else:
            self.file_path = None
        self.chunk_size = chunk_size

        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        # extra info about the current episode, that was passed in during
        # reset()
        self.current_reset_info = {}
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
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If "
                               "you want to allow early resets, "
                               "wrap your env with Monitor(env, path, "
                               "allow_early_resets=True)")
        self.rewards = []
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError(
                    'Expected you to pass kwarg %s into reset' % key)
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Step the environment with the given action

        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward,
                                                           done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            eplen = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": eplen,
                       "t": round(time.time() - self.t_start, 6)}
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_rewards.append(ep_rew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            self.data.append(ep_info)
            if len(self.data) >= self.chunk_size:
                self.save()
            info['episode'] = ep_info
        self.total_steps += 1
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
        return self.episode_rewards

    def get_episode_lengths(self):
        """
        Returns the number of timesteps of all the episodes

        :return: ([int])
        """
        return self.episode_lengths

    def get_episode_times(self):
        """
        Returns the runtime in seconds of all the episodes

        :return: ([float])
        """
        return self.episode_times
