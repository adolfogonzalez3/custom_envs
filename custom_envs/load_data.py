
import numpy as np

#import mnist

def load_data():
    return np.load('/home/adolfogonzaleziii/Documents/custom_envs/custom_envs/iris.npz')['data']
