#!/usr/bin/env python3
"""
This module makes predictions on a given
dataset using a trained Keras model.
"""
import tensorflow as tf


def predict(model, X, verbose=0):
    """
    Save only the weights of a trained Keras model.

    Args:
        model: A trained Keras model.
        X: Input data with a shape of (number of examples, input features).
        verbose: (Optional) Verbosity level during predictions:
        ´        0: Silent (default).
                 1: Displays a progress bar.
                 2: Displays one line per batch.
    Returns:
        predictions: A list of predicted class labels for the input data.
    """
    probalilities = model.predict(X, verbose=verbose)
    predictions = tf.argmax(probalilities, axis=1)

    return predictions
