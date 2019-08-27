'''Module for evaluating learned agents against different environments.'''
import os
import argparse
from pathlib import Path
from functools import partial
from itertools import chain

import numpy as np
import pandas as pd
import tensorflow as tf
import optuna

from custom_envs.data import load_data
import custom_envs.utils.utils_file as utils_file


def flatten_arrays(arrays):
    return list(chain.from_iterable(a.ravel().tolist() for a in arrays))


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, trial, target='loss'):
        super().__init__()
        self.epoch = None
        self.history = None
        self.trial = trial
        self.target = target

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        self.history.append({
            'epoch': epoch,
            'weights_mean': np.mean(flatten_arrays(self.model.get_weights())),
            **logs
        })
        self.trial.report(logs[self.target], epoch)
        if self.trial.should_prune():
            raise optuna.structs.TrialPruned()


def train_model(parameters, trial):
    '''Train an model to classify a data set.'''
    parameters = parameters.copy()
    path = Path(parameters['path'], str(trial.number))
    parameters['path'] = str(path)
    batch_size = trial.suggest_categorical('batch_size',
                                           [2**i for i in range(5, 12)])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e0)
    parameters.update({
        'batch_size': batch_size, 'learning_rate': learning_rate
    })
    sequence = load_data(parameters['data_set'])
    features = sequence.features
    labels = sequence.labels
    layers = [features.shape[-1], labels.shape[-1]]
    use_bias = True
    graph = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with graph.as_default():
        target = tf.keras.layers.Input((1,))
        model_in = tf.keras.layers.Input([layers[0]])
        tensor = model_in
        layers = [layers[0], layers[-1]]
        for layer in layers[1:-1]:
            layer = tf.keras.layers.Dense(layer, use_bias=use_bias,
                                          activation='relu')
            tensor = layer(tensor)
        layer = tf.keras.layers.Dense(layers[-1], use_bias=use_bias,
                                      activation='softmax')
        tensor = layer(tensor)
        model = tf.keras.Model(inputs=model_in, outputs=tensor)
        callbacks = [CustomCallback(trial, 'loss')]
        with tf.Session(graph=graph, config=config):
            try:
                model.compile(tf.train.AdamOptimizer(learning_rate),
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
                model.fit(features, labels, epochs=parameters['total_epochs'],
                          shuffle=True, batch_size=batch_size, verbose=0,
                          callbacks=callbacks)
            finally:
                path.mkdir()
                utils_file.save_json(parameters, path / 'parameters.json')
                dataframe = pd.DataFrame.from_dict(callbacks[0].history)
                dataframe.to_csv(path / 'results.csv')
                model.save(str(path / 'model.hdf5'))
    return dataframe['loss'].iloc[-1]


def main():
    '''Evaluate a trained model against logistic regression.'''
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="The path to save the results.")
    parser.add_argument("--trials", help="The number of trials.",
                        type=int, default=10)
    parser.add_argument("--total_epochs", help="The number of epochs.",
                        type=int, default=40)
    parser.add_argument("--data_set", help="The data set to trial against.",
                        type=str, default='iris')
    args = parser.parse_args()
    parameters = vars(args).copy()
    del parameters['trials']
    path = Path(parameters['path'])
    if not path.exists():
        path.mkdir()
        utils_file.save_json(parameters, path / 'parameters.json')
    else:
        if (path / 'study.db').exists():
            print('Directory exists. Using existing study and parameters.')
            parameters = utils_file.load_json(path / 'parameters.json')
        else:
            raise FileExistsError(('Directory already exists and is not a '
                                   'study.'))
    objective = partial(train_model, parameters)
    storage = 'sqlite:///' + str(path / 'study.db')
    study = optuna.create_study(study_name=str(path.name), storage=storage,
                                load_if_exists=True,
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=args.trials)


if __name__ == '__main__':
    main()
