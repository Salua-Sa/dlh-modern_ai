#!/usr/bin/env python3
"""
This module returns a configured gradient descent optimizer and the
appropriate batch size based on the selected gradient descent variant.
"""

from tensorflow import keras


def train_with_gradient_descent_variant(variant,
                                        learning_rate, x_train, batch_size):
    """
    Return a configured gradient descent optimizer and the appropriate
    batch size based on the selected gradient descent variant.

    Args:
        variant: (str) The training variants:
        - batch: Perform updates based on the entire dataset.
        - stochastic: Perform updates on a single training example at a time.
        - mini_batch: Perform updates on a custom-sized batch of training
          examples.
        learning_rate: (float) The learning rate for the optimizer.
        x_train: The training dataset (input data).
        batch_size: (int) The batch size to use when 'mini_batch' is selected.

    Returns:
        optimizer: A Gradient Descent optimizer configured with
                   the specified learning rate.
        bs: The correct batch size based on the selected variant.
    """
    optimizer = keras.optimizers.SGD(
        learning_rate=learning_rate
        )

    if variant == "batch":
        bs = x_train.shape[0]
    elif variant == "stochastic":
        bs = 1
    elif variant == "mini_batch":
        bs = batch_size
    else:
        raise ValueError("Invalid variant")

    return optimizer, bs
