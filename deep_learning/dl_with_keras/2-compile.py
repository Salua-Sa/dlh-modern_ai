#!/usr/bin/env python3
"""This module configures the keras model for training.
"""
from tensorflow import keras


def compile_model(model, learning_rate=0.01):
    """
    Configure the keras model for training having:
    - Stochastic gradient descent as the optimizer.
    - Binary cross-entropy loss as the loss function.
    - Include accuracy as a metric to monitor classification performance.

    Args:
        model: keras model.
        learning_rate: Learning rate for
            gradient descent (default is 0.01).

    Returns:
        None
    """
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'])
