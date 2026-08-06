#!/usr/bin/env python3
"""This module saves and reloads only
the weights of a trained Keras model.
"""


def save_model_weights(model, filepath):
    """
    Save only the weights of a trained Keras model.

    Args:
        model: A trained Keras model whose weights need to be saved.
        filepath: A string representing the file path
                 (including the file name) where the model will be saved.
    Returns:
        None. The function saves the model to the specified location.
    """
    model.save_weights(filepath)


def load_model_weights(model, filepath):
    """
    Reload a Keras model.
    Args:
        model: A compatible Keras model instance where
               the weights will be loaded.
        filepath: A string representing the file path
        (including the file name) from where the model will be loaded.

    Returns:
        None. The function loads the weights into the provided model.
    """
    model.load_weights(filepath)
