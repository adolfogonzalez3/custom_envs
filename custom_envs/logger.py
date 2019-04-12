import time
import os
import numpy as np
import numpy.random as npr


def rand():
    return int.from_bytes(npr.bytes(8), 'big')

class Logger:
    def __init__(self, environment, seed=None):
        self.log_id = str(rand())
        self.environment = environment
        self.path = '/home/adolfogonzaleziii/Documents/custom_envs/parts/'
        self.data = []

    def save(self):
        filename = os.path.join(self.path, str(rand()))
        print(filename)
        np.savez_compressed(filename, environment=self.environment,
                            log_id=self.log_id, data=self.data)

    def write(self, currentstep, reward, accuracy, weightmag):
        self.data.append([currentstep, reward, accuracy, weightmag])
        if len(self.data) > 128:
            self.save()
            self.data = []

    def close(self):
        self.save()
        self.data = []


def task(args):
    print(args)
    test = Logger('huh?', 0)
    print(test.log_id)
    for i in range(2000):
        test.write(i, .5, .3, .3)
    test.close()

if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor, wait

    with ProcessPoolExecutor() as executor:
        fut = executor.map(task, [i for i in range(10)])
    fut = list(fut)
    #print(Logger.get_results())
    #print(Logger.get_experiments())
    #print(len(Logger.get_experiments()))