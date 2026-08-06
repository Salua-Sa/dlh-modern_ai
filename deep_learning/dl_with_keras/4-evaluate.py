#!/usr/bin/env python3
"""This module assess a trained Keras model's performance.
"""


def evaluate_model(model, X, Y, verbose=0):
    """
    Assess a trained Keras model's performance on a given data.

    Args:
        model: A trained Keras model.
        X: Input data with a shape of (number
           of examples, input features).
        Y: True labels corresponding to the input data
           with a shape of (number of examples, 1).
        verbose: Verbosity mode (0 = silent,
           1 = progress bar).
    Returns:
        loss: The calculated loss on the provided data.
        accuracy: The accuracy of the model on the provided data.
    """
    loss, accuracy = model.evaluate(X, Y, verbose=verbose)

    return (loss, accuracy)
