'''Module for creating vectorized environment runners.'''
import multiprocessing as mp
from threading import Thread
from collections import OrderedDict

import gym
import numpy as np

from stable_baselines.common.vec_env import VecEnv, CloudpickleWrapper
from stable_baselines.common.tile_images import tile_images


def create_pipe():
    return mp.Pipe(True)


def get_context(start_method):
    '''Get a new multiprocessing context.'''
    if start_method is None:
        # Use thread safe method, see issue #217.
        # forkserver faster than spawn but not always available.
        forkserver_available = 'forkserver' in mp.get_all_start_methods()
        start_method = 'forkserver' if forkserver_available else 'spawn'
    return mp.get_context(start_method)


def _worker(remote, env_fn_wrapper):
    env = env_fn_wrapper.var()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                if any(done):
                    # save final observation where user can get it, then reset
                    #info['terminal_observation'] = observation
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'render':
                remote.send(env.render(*data[0], **data[1]))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError
    except EOFError:
        pass
    finally:
        env.close()


class ConcurrentVecEnv(VecEnv):
    """
    Creates a concurrent vectorized wrapper for multiple environments

    :param env_fns: ([Gym Environment]) Environments to run in subprocesses
    :param create_method: (str) method used to start the concurrent units.
                                Should create an object which implements
                                python's Thread api.
    """

    def __init__(self, env_fns, create_method):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        self.remotes, self.work_remotes = zip(*[create_pipe()
                                                for _ in range(n_envs)])
        self.processes = []
        for work_remote, env_fn in zip(self.work_remotes, env_fns):
            args = (work_remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause
            # things to hang
            process = create_method(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return (_flatten_obs(obs, self.observation_space), np.stack(rews),
                np.stack(dones), infos)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return _flatten_obs(obs, self.observation_space)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def render(self, *args, **kwargs):
        mode = kwargs.get('mode', 'human')
        kwargs['mode'] = 'rgb_array'
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(('render', (args, kwargs)))
        imgs = [pipe.recv() for pipe in self.remotes]
        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        for pipe in self.remotes:
            pipe.send(('render', {"mode": 'rgb_array'}))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def env_method(self, method_name, *method_args, **method_kwargs):
        """
        Call an arbitrary class methods on a vectorized environment.

        :param method_name: (str) The name of the env class method to invoke
        :param method_args: (tuple) Any positional arguments to provide in the
                                    call
        :param method_kwargs: (dict) Any keyword arguments to provide in the
                                     call
        :return: (list) List of items retured by each environment's method call
        """

        for remote in self.remotes:
            remote.send(
                ('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in self.remotes]

    def get_attr(self, attr_name):
        """
        Provides a mechanism for getting class attribues from vectorized environments
        (note: attribute value returned must be picklable)

        :param attr_name: (str) The name of the attribute whose value to return
        :return: (list) List of values of 'attr_name' in all environments
        """

        for remote in self.remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in self.remotes]

    def set_attr(self, attr_name, value, indices=None):
        """
        Provides a mechanism for setting arbitrary class attributes inside vectorized environments
        (note:  this is a broadcast of a single value to all instances)
        (note:  the value must be picklable)

        :param attr_name: (str) Name of attribute to assign new value
        :param value: (obj) Value to assign to 'attr_name'
        :param indices: (list,tuple) Iterable containing indices of envs whose attr to set
        :return: (list) in case env access methods might return something, they will be returned in a list
        """

        if indices is None:
            indices = range(len(self.remotes))
        elif isinstance(indices, int):
            indices = [indices]
        for remote in [self.remotes[i] for i in indices]:
            remote.send(('set_attr', (attr_name, value)))
        return [remote.recv() for remote in [self.remotes[i] for i in indices]]


def _flatten_obs(obs, space):
    """
    Flatten observations, depending on the observation space.

    :param obs: (list<X> or tuple<X> where X is dict<ndarray>, tuple<ndarray> or ndarray) observations.
                A list or tuple of observations, one per environment.
                Each environment observation may be a NumPy array, or a dict or tuple of NumPy arrays.
    :return (OrderedDict<ndarray>, tuple<ndarray> or ndarray) flattened observations.
            A flattened NumPy array or an OrderedDict or tuple of flattened numpy arrays.
            Each NumPy array has the environment index as its first axis.
    """
    assert isinstance(
        obs, (list, tuple)), "expected list or tuple of observations per environment"
    assert len(obs) > 0, "need observations from at least one environment"

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(
            space.spaces, OrderedDict), "Dict space must have ordered subspaces"
        assert isinstance(
            obs[0], dict), "non-dict observation for environment with Dict observation space"
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(
            obs[0], tuple), "non-tuple observation for environment with Tuple observation space"
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)


class SubprocVecEnv(ConcurrentVecEnv):
    """
    Creates a multiprocess vectorized wrapper for multiple environments

    :param env_fns: ([Gym Environment]) Environments to run in subprocesses
    :param start_method: (str) method used to start the subprocesses.
           Must be one of the methods returned by
           multiprocessing.get_all_start_methods(). Defaults to 'forkserver' on
           available platforms, and 'spawn' otherwise. Both 'forkserver' and
           'spawn' are thread-safe, which is important when TensorFlow sessions
           or other non thread-safe libraries are used in the parent
           (see issue #217). However, compared to 'fork' they incur a small
           start-up cost and have restrictions on global variables.
           For more information, see the multiprocessing documentation.
    """

    def __init__(self, env_fns, start_method=None):
        ctx = get_context(start_method)
        super().__init__(env_fns, ctx.Process)


class ThreadVecEnv(ConcurrentVecEnv):
    """
    Creates a threaded vectorized wrapper for multiple environments

    :param env_fns: ([Gym Environment]) Environments to run in subprocesses
    :param start_method: (str) method used to start the subprocesses.
           Must be one of the methods returned by
           multiprocessing.get_all_start_methods(). Defaults to 'forkserver' on
           available platforms, and 'spawn' otherwise. Both 'forkserver' and
           'spawn' are thread-safe, which is important when TensorFlow sessions
           or other non thread-safe libraries are used in the parent
           (see issue #217). However, compared to 'fork' they incur a small
           start-up cost and have restrictions on global variables.
           For more information, see the multiprocessing documentation.
    """

    def __init__(self, env_fns):
        super().__init__(env_fns, Thread)
