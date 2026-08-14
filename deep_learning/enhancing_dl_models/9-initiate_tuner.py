#!/usr/bin/env python3
"""
This module initializes a Keras Tuner for hyperparameter tuning.
"""
import keras_tuner


def initiate_tuner(tuner_type, build_model, x_train,
                   y_train, seed, hyperband_iterations, max_trials,
                   objective="val_accuracy", overwrite=True):
    """
    Initialize a Keras Tuner for hyperparameter tuning.
    Not allowed to import any module except import keras_tuner

    Args:
        tuner_type: (str) Type of tuner. Must be one of 'Hyperband',
                    'RandomSearch', or 'BayesianOptimization'.
        build_model: (function) A function that returns a
                     compiled Keras model.
        x_train: (ndarray) Training features.
        y_train: (ndarray) Training labels.
        seed: (int) The random seed.
        hyperband_iterations: (int) Number of iterations for Hyperband tuning.
        max_trials: (int) Maximum number of trials
                    for RandomSearch and BayesianOptimization.
        objective: (str) Metric to optimize during tuning.
        overwrite: (bool) Whether to overwrite the previous tuning project.
                   Default is True.

    Returns:
        A Keras Tuner object (Hyperband, RandomSearch, or
        BayesianOptimization) ready for use in hyperparameter optimization.
    """
    if tuner_type == "Hyperband":
        tuner = keras_tuner.Hyperband(
            hypermodel=build_model,
            objective=objective,
            max_epochs=10,
            factor=3,
            hyperband_iterations=hyperband_iterations,
            seed=seed,
            directory='dir_tuner',
            project_name='hyperband',
            overwrite=overwrite
            )
    elif tuner_type == "RandomSearch":
        tuner = keras_tuner.RandomSearch(
            hypermodel=build_model,
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            directory='dir_tuner',
            project_name='random_search',
            overwrite=overwrite
            )
    elif tuner_type == "BayesianOptimization":
        tuner = keras_tuner.BayesianOptimization(
            hypermodel=build_model,
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            directory='dir_tuner',
            project_name='bayesian_optimization',
            overwrite=overwrite
            )
    else:
        raise ValueError("Invalid tuner type")

    return tuner
