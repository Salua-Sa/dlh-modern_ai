#!/usr/bin/env python3
"""
This module returns a configured SGD-based optimizer based
on the specified variant, including options for momentum
and Nesterov acceleration.
"""
from tensorflow import keras


def get_optimizer_SGD(name, lr, momentum=0.0, nesterov=False):
    """
    Return a configured SGD-based optimizer based on the
    specified variant, including options for momentum and
    Nesterov acceleration.

    Args:
        name: (str) the optimizer variant:
        - SGD: Standard stochastic gradient descent.
        - SGD+Momentum: SGD with classical momentum.
        - SGD+Momentum+Nesterov: SGD with momentum and Nesterov acceleration.
        lr: (float) The learning rate.
        momentum: (float) The momentum factor.
        nesterov: (boolean) Indicating whether to apply Nesterov acceleration
        (default is False).

    Returns:
        optimizer: A Keras SGD optimizer instance configured with
        the provided settings.
    """
    if name == "SGD":
        optimizer = keras.optimizers.SGD(
            learning_rate=lr,
            momentum=0.0,
            nesterov=False)
    elif name == "SGD+Momentum":
        optimizer = keras.optimizers.SGD(
            learning_rate=lr,
            momentum=momentum,
            nesterov=False)
    elif name == "SGD+Momentum+Nesterov":
        optimizer = keras.optimizers.SGD(
            learning_rate=lr,
            momentum=momentum,
            nesterov=False)
    else:
        raise ValueError("Invalid name")

    return optimizer
