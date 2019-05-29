
import numpy as np
from PIL import Image

def resize_image(array, shape):
    '''
    
    '''
    return np.asarray(Image.fromarray(array).resize(shape))

def resize_all(array):
    return np.array([resize_image(sample.reshape((28, 28)), (7, 7)).ravel()
                     for sample in array])