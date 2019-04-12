
class MultiAgent:
    def __init__(self, agent_fns, env, seed=None):
        env = gym.make(env_name)
        env.seed(seed)
        self.envs = env
        self.agents = [Thread(fn)]
