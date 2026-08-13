#!/usr/bin/env python3
"""
This module creates a customizable early stopping
callback for Keras training.
"""
from tensorflow import keras


def get_early_stopping_callback(patience, monitor='val_loss', verbose=1):
    """
    Create a customizable early stopping callback for Keras training.
    This callback should:
    - Monitor a specific metric during training.
    - Stop training if no improvement is seen after a defined number of epochs.
    - Must restore the best model weights once training stops.

    Args:
        patience: (int) Number of epochs to wait without improvement
                  before stopping training.
        monitor: (str) Metric to monitor, such as val_loss or val_accuracy.
        verbose: (int) Verbosity mode to display messages when
                 the callback takes an action.

    Returns:
        keras.callbacks.EarlyStopping.
    """
    callback = keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        verbose=verbose,
        restore_best_weights=True
        )

    return callback
