#!/usr/bin/env python3
"""
This module performs hyperparameter tuning
and retrieve the best hyperparameters.
"""


def search_and_return_best_model(tuner, x_train, y_train,
                                 epochs, validation_split, verbose=0):
    """
    Perform hyperparameter tuning and retrieve the best hyperparameters.

    Args:
        tuner: A Keras Tuner object (Hyperband, RandomSearch,or
               BayesianOptimization) that wraps the hyperparameter
               search process.
        x_train: (ndarray) Training input data.
        y_train: (ndarray) Training target data.
        epochs: (int) Number of training epochs for each trial during
                the search.
        validation_split: (float) Fraction of training data to use as
                          validation during tuning.
        verbose: Verbosity mode(0 = silent, 1 = search bar)

    Returns:
        best_hyperparameters: The hyperparameter configuration
        that led to the best model, as a
        kerastuner.engine.hyperparameters.HyperParameters object.
    """
    tuner.search(
        x=x_train,
        y=y_train,
        epochs=epochs,
        validation_split=validation_split,
        verbose=verbose
        )
    best_hyperparameters = tuner.get_best_hyperparameters(num_trieals=1)[0]

    return best_hyperparameters
