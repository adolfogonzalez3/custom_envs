
import numpy as np
import numpy.random as npr

import tensorflow.keras as keras
import tensorflow.keras.layers as k_layers

    

def main():
    target = k_layers.Input((1,))
    model = keras.Sequential()
    model.add(k_layers.Dense(10, input_shape=(3,), use_bias=False))
    loss_obj = keras.losses.CategoricalCrossentropy()
    loss = loss_obj(target, model.output)
    grads = keras.backend.gradients(loss, model.weights)
    print(model.weights)
    print(model.input, model.output, target, loss)
    print(grads)
    get_grads = keras.backend.function((model.input, target), grads)
    
    

    x = np.ones((32, 3))
    y = np.ones((32, 1))

    grad = get_grads((x, y))

    print(grad)
    print(model.weights)
    print([npr.normal(size=w.shape.as_list()) for w in model.weights])
    model.set_weights([npr.normal(size=w.shape.as_list()) for w in model.weights])
    model.summary()

if __name__ == '__main__':
    main()
