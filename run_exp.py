
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from itertools import product

ENV_NAMES = ('Optimize-v0', 'OptLR-v0', 'OptLRs-v0')

import gym
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.misc_util import set_global_seeds



def task2(args):
    print(args)
    seed, path, env_name = args
    log_dir = os.path.join(path, env_name, 'ppo-{:d}'.format(seed))
    save_path = os.path.join(path, env_name, 'ppo-{:d}'.format(seed), 'model')
    os.makedirs(log_dir, exist_ok=True)
    env = gym.make(env_name)
    env.seed(seed)
    set_global_seeds(seed)
    env = Monitor(env, log_dir, allow_early_resets=True,
                  info_keywords=('objective','accuracy'))
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

    model = PPO2(MlpPolicy, env, verbose=0)
    model.learn(total_timesteps=10**7)
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


if __name__ == '__main__':
    import argparse
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument("save_folder", help="The directory to save files to.")
    ARGS = PARSER.parse_args()
    PATH = [ARGS.save_folder]
    task2((0, PATH[0], ENV_NAMES[0]))
    #with ProcessPoolExecutor() as executor:
    #    executor.map(task2, product(range(100), PATH, ENV_NAMES))
