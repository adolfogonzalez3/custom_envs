
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer


def build_perceptron(shape, n_class):
    model = Sequential()
    model.add(InputLayer(shape))
    model.add(Dense(n_class, activation='softmax'))
    model.build()
    return model
