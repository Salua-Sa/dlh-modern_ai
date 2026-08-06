#!/usr/bin/env python3
"""This module saves and reloads a Keras model.
"""
from tensorflow import keras


def save_model(model, filepath):
    """
    Save a Keras model, including its architecture, weights, and optimizer state.

    Args:
        model: A trained Keras model to be saved.
        filepath: A string representing the file path
                 (including the file name) where the model will be saved.
    Returns:
        None. The function saves the model to the specified location.
    """
    model.save(filepath)


def load_model(filepath):
    """
    Reload a Keras model.
    Args:
        filepath: A string representing the file path
        (including the file name) from where the model will be loaded.

    Returns:
        model: The reloaded Keras model.
    """
    model = keras.models.load_model(filepath)

    return model
