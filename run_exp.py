
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from itertools import product

ENV_NAMES = ('Optimize-v0', 'OptLR-v0', 'OptLRs-v0')

import numpy as np
import gym
from stable_baselines.bench import Monitor
import stable_baselines.ddpg as ddpg
from stable_baselines.common.policies import MlpPolicy, MlpLnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2, A2C, DDPG
from stable_baselines.common.misc_util import set_global_seeds



def task2(args):
    print(args)
    seed, path, env_name = args
    log_dir = os.path.join(path, env_name, 'ppo-{:d}'.format(seed))
    save_path = os.path.join(path, env_name, 'ppo-{:d}'.format(seed), 'model')
    os.makedirs(log_dir)
    env = gym.make(env_name)
    env.seed(seed)
    set_global_seeds(seed)
    env = Monitor(env, log_dir, allow_early_resets=True,
                  info_keywords=('objective','accuracy'))
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

    model = PPO2(MlpPolicy, env, verbose=0)
    model.learn(total_timesteps=10**6)
    model.save(save_path)
    
def task_hyperparam(args):
    print(args)
    seed, path, env_name, learning_rate, gamma, alg = args
    task_name = '{}_lr_{:.4f}_g_{:.4f}-{:d}'.format(alg, learning_rate, gamma,
                                                    seed)
    log_dir = os.path.join(path, env_name, task_name)
    save_path = os.path.join(path, env_name, task_name, 'model')
    os.makedirs(log_dir)
    env = gym.make(env_name)
    env.seed(seed)
    set_global_seeds(seed)
    env = Monitor(env, log_dir, allow_early_resets=True,
                  info_keywords=('objective','accuracy'))
    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: env])

    if alg == 'PPO':
        model = PPO2(MlpPolicy, env, gamma=gamma, learning_rate=learning_rate,
                     verbose=0)
    elif alg == 'A2C':
        model = A2C(MlpPolicy, env, gamma=gamma, learning_rate=learning_rate,
                     verbose=0)
    else:
        model = DDPG(ddpg.MlpPolicy, env, gamma=gamma,  verbose=0,
                     actor_lr=learning_rate/10, critic_lr=learning_rate)
    model.learn(total_timesteps=10**6)
    model.save(save_path)

def task(args):
    """Run an environment on PPO.
    
    :param args: A tuple that contains the seed, path to save model and results
                 and the name of the environment.
    """
    print(args)
    seed, path, env_name = args
    env = os.environ
    env['OPENAI_LOGDIR'] = os.path.join(path, env_name,
                                        'ppo-{:d}'.format(seed))
    save_path = os.path.join(path, env_name, 'ppo-{:d}'.format(seed), 'model')
    subprocess.run(['python', '-m', 'stable_baselines.run', '--alg=ppo2',
                    '--seed={:d}'.format(seed),
                    '--env={}'.format(env_name),# '--network=lstm',
                    '--num_timesteps=2e4',# '--num_env=4',
                    '--save_path={}'.format(save_path)],
                   env=env, check=True)


def main_test_all_envs():
    import argparse
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("save_folder", help="The directory to save files to.")
    PARSER.add_argument('--num_of_seeds', help='Number of seeds to run',
                        default=10)
    ARGS = PARSER.parse_args()
    PATH = [ARGS.save_folder]
    
    with ProcessPoolExecutor() as executor:
        job_details = product()
        executor.map(task2, product(range(ARGS.num_of_seeds), PATH, ENV_NAMES))

def main_hyperparams():
    import argparse
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("save_folder", help="The directory to save files to.")
    PARSER.add_argument('--num_of_seeds', help='Number of seeds to run',
                        default=20)
    ARGS = PARSER.parse_args()
    SEEDS = list(range(ARGS.num_of_seeds))
    PATH = [ARGS.save_folder]
    ENVS = ['Optimize-v0']
    LEARNING_RATES = np.linspace(1, 1e-3, 10)
    GAMMAS = np.linspace(1, 0, 10, False)
    ALGS = ['PPO', 'A2C', 'DDPG']
    task_details = product(SEEDS, PATH, ENVS, LEARNING_RATES, GAMMAS, ALGS)
    
    with ProcessPoolExecutor() as executor:
        executor.map(task_hyperparam, task_details)


if __name__ == '__main__':
    main_hyperparams()