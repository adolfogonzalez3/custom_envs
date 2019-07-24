'''Module for evaluating learned agents against different environments.'''
import os
import argparse
from pathlib import Path
from functools import partial
from itertools import chain

import optuna
import numpy as np
import pandas as pd
import tensorflow as tf

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
    graph = tf.Graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with graph.as_default():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(
            48, input_shape=features.shape[1:],
            kernel_initializer=tf.keras.initializers.glorot_normal(),
            bias_initializer=tf.keras.initializers.glorot_normal(),
            activation='relu', use_bias=True
        ))
        model.add(tf.keras.layers.Dense(
            48, activation='relu', use_bias=True,
            kernel_initializer=tf.keras.initializers.glorot_normal(),
            bias_initializer=tf.keras.initializers.glorot_normal()
        ))
        model.add(tf.keras.layers.Dense(
            labels.shape[-1], activation='softmax',
            kernel_initializer=tf.keras.initializers.glorot_normal(),
            bias_initializer=tf.keras.initializers.glorot_normal()
        ))
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
    parser.add_argument("path", help="The path to save the results.",
                        type=Path)
    parser.add_argument("--trials", help="The number of trials.",
                        type=int, default=10)
    parser.add_argument("--total_epochs", help="The number of epochs.",
                        type=int, default=40)
    parser.add_argument("--data_set", help="The data set to trial against.",
                        type=str, default='iris')
    args = parser.parse_args()
    parameters = vars(args)
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
    # print(study.best_trial.number)
    #dataframe = study.trials_dataframe()
    # print(dataframe)
    # print(dataframe.columns)
    # print(dataframe['number'])
    # print(dataframe.loc[('number',)])
    # print(dataframe.columns)
    #dataframe = dataframe[dataframe['params']['batch_size'] == 32]
    # print(dataframe)
    # print(dataframe['intermediate_values'])


if __name__ == '__main__':
    main()
