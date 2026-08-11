#!/usr/bin/env python3
"""
This module returns a Keras optimizer configured based
on the specified optimizer name and its corresponding parameters.
"""
from tensorflow import keras


def get_optimizer(name, learning_rate, momentum, beta_1, beta_2, rho):
    """
    Return a Keras optimizer configured based on the specified
    optimizer name and its corresponding parameters.

    Args:
        name: (str) The name of the optimizer to use :
        - sgd: Stochastic Gradient Descent, with optional momentum.
        - adam: Adaptive Moment Estimation: combines the benefits
                of RMSprop and momentum-based optimization techniques.
        - rmsprop: Root Mean Square Propagation: adapts the learning
                   rate for each parameter based on its historical gradients.
        learning_rate: (float) The learning rate for the optimizer.
        momentum: (float) The momentum factor (only used for SGD).
        beta_1: (float) The exponential decay rate for the first moment
                estimate (only used for Adam).
        beta_2: (float) The exponential decay rate for the second moment
                estimate (only used for Adam).
        rho: (float) The decay factor for RMSprop (only used for RMSprop).

    Returns:
        optimizer: A Keras optimizer instance (SGD, Adam, or RMSprop)
                   configured with the provided settings.
    """
    if name == "sgd":
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum
            )
    elif name == "adam":
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2
            )
    elif name == "rmsprop":
        optimizer = keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            rho=rho
            )
    else:
        raise ValueError("Invalid name")

    return optimizer
