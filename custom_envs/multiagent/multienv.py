
import gym

class MultiEnvServer:
    def __init__(self, env, slices):
        assert(np.sum(env.action_space.shape) == np.sum(slices))
        self.env = env
        



class MultiEnv:
    def __init__(self, queue, action_slice):
        self.queue = queue

    def step(self, action):
        self.queue.submit_action(action)
        return self.queue.retrieve_step_tuple()

    def reset

if __name__ == '__main__':
    env_name = 'Optimize-v0'
    env = gym.make(env_name)
    print(env.observation_space.shape)
    print(env.action_space.shape)