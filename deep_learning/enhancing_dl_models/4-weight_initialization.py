#!/usr/bin/env python3
"""
This module returns a compiled Keras model.
"""
from tensorflow import keras


def build_model_initializer_by_activation(input_dim, hidden_units, activation):
    """
    Return a compiled Keras model with:
    - One hidden layer that uses an appropriate weight initializer
    based on the activation function.
    - Followed by a softmax output layer.

    Args:
        input_dim: (int) The number of input features.
        hidden_units: (int) The number of neurons in the hidden layer.
        activation: (string) the activation function to use
                    in the hidden layer:
        - sigmoid and tanh: Use Glorot Uniform initializer.
        - relu and leaky_relu: Use He Normal initializer.

    Returns:
        model: A Keras model with the described architecture.
    """
    if activation == "sigmoid" or activation == "tanh":
        initializer = keras.initializers.GlorotUniform()
        acttivation_func = activation
    elif activation == "relu":
        initializer = keras.initializers.HeNormal()
        acttivation_func = activation
    elif activation == "leaky_relu":
        initializer = keras.initializers.HeNormal()
        acttivation_func = keras.layers.LeakyRelu()
    else:
        raise ValueError("invalid activation")

    inputs = keras.Input(shape=(input_dim,))

    hidden = keras.layers.Dense(
        units=hidden_units,
        activation=acttivation_func,
        kernel_initializer=initializer,
        )(inputs)

    outputs = keras.layers.Dense(
        units=10,
        activation="softmax"
        )(hidden)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        )

    return model
