#!/usr/bin/env python3
"""
This module creates a Keras model with L2 regularization.
"""
from tensorflow import keras


def build_model_with_L2_regularization(input_dim, hidden_units,
                                       n_layers, lambda_l2):
    """
    Create a Keras model with L2 regularization:
    Multiple hidden layers, each consisting of:
    A dense layer.
    ReLU activation.
    L2 regularization applied to the kernel weights.
    Followed by a softmax output layer.

    Args:
        input_dim: (int) The number of input features.
        hidden_units: (int) The number of neurons in each hidden layer.
        n_layers: (int) specifying the number of hidden layers to include.
        lambda_l2: (float) The strength of L2 regularization.

    Returns:
        model: A Keras model with the described architecture
               and L2 regularization.
    """
    inputs = keras.Input(shape=(input_dim,))

    x = inputs
    for i in range(n_layers):
        x = keras.layers.Dense(
            units=hidden_units,
            activation='relu',
            kernel_regularizer=keras.regularizers.L2(lambda_l2)
            )(x)

    outputs = keras.layers.Dense(
        units=10,
        activation='softmax'
        )(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        )

    return model
