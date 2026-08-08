#!/usr/bin/env python3
"""
This module logs a Keras model’s training metrics to TensorBoard.
"""
from tensorflow import keras
import datetime


def log_to_tensorboard(log_dir, model, X, Y, epochs, verbose=1):
    """
    Log a Keras model’s training metrics to TensorBoard.
    TensorBoard callback configure:
    - Logs training metrics (e.g., loss and accuracy) after each epoch.
    - Logs weight histograms and activation histograms using
    histogram_freq=1 to help visualize how weights evolve over time.
    - Saves logs in a subdirectory named with a unique timestamp in
    the format YYYYMMDD-HHMMSS (e.g. 20250616-153245) to prevent
    overwriting logs from previous runs.

    Args:
        log_dir: (str) Base directory where logs should be saved.
        model: Keras model.
        X: Input data, shape (number of examples, input features).
        Y: labels, shape (number of examples, 1).
        epochs: Number of training epochs.
        verbose: Verbosity mode (0 = silent, 1 = progress bar).

    Returns:
        None
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = f"{log_dir}/{timestamp}"
    callback = keras.callbacks.TensorBoard(
        log_dir=path,
        histogram_freq=1
        )
    model.fit(X, Y, epochs=epochs, verbose=verbose, callbacks=[callback])
