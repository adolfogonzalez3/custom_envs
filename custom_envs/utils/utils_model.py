@wrap_in_session
def create_conv_net(input_shape, output_size, kernel_sizes=(3, 3),
                    filter_sizes=(32, 64), layers=(256, 256),
                    activation='relu'):
    '''
    Create a wrapped keras convnet with its own private session.

    :param input_shape: (Sequence) The shape of the expected input.
    :param output_size: (int) The number of labels intended to be predicted.
    :param kernel_sizes: (Sequence) Defines the sizes of the kernels.
    :param filter_sizes: (Sequence) Defines the number of filters.
    :param layers: (Sequence) Defines the number of hidden layers.
    :param activations: (str) Defines the activation function to use.
    :return: (WrappedSession(tf.keras.Model)) A keras model
    '''
    model = Sequential()
    model.add(InputLayer(input_shape))
    for k_size, f_size in zip(kernel_sizes, filter_sizes):
        model.add(Conv2D(
            f_size, kernel_size=k_size, activation=activation, padding='same'
        ))
        model.add(MaxPooling2D(2))
    model.add(Flatten())
    for hidden_units in layers:
        model.add(Dense(hidden_units, activation=activation))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(
        optimizer="adam", loss='binary_crossentropy', metrics=['accuracy']
    )
    return model


@wrap_in_session
def create_neural_net(input_shape, output_size, layers=(256, 256),
                      activation='relu'):
    '''
    Create a wrapped keras neural network with its own private session.

    :param input_shape: (Sequence) The shape of the expected input.
    :param output_size: (int) The number of labels intended to be predicted.
    :param layers: (Sequence) Defines the number of hidden layers.
    :param activations: (str) Defines the activation function to use.
    :return: (WrappedSession(tf.keras.Model)) A keras model
    '''
    model = Sequential()
    model.add(InputLayer(input_shape))
    if len(input_shape) > 1:
        model.add(Flatten())
    for hidden_units in layers:
        model.add(Dense(hidden_units, activation=activation))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(
        optimizer="adam", loss='binary_crossentropy', metrics=['accuracy']
    )
    return model