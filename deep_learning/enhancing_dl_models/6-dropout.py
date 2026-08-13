#!/usr/bin/env python3
"""
This module creates a Keras model with dropout regularization.
"""
from tensorflow import keras


def build_model_with_dropout(input_dim, hidden_units,
                             n_layers, dropout_rate_input,
                             dropout_rate_hidden):
    """
    Create a Keras model with dropout regularization:
    The architecture should include:
    - An input layer followed by a dropout layer.
    - Multiple hidden layers, each consisting of:
       - A dense layer.
       - ReLU activation.
       - A dropout layer applied after each hidden layer.
    - A final output layer with softmax activation for classification.

    Args:
        input_dim: (int) Number of input features.
        hidden_units: (int) Number of neurons in each hidden layer.
        n_layers: (int) Number of hidden layers to include.
        dropout_rate_input: (float) Dropout rate to apply after
                            the input layer.
        dropout_rate_hidden: (float) Dropout rate to apply after
                             each hidden layer.

    Returns:
        model: A Keras model instance with the described architecture.
    """
    inputs = keras.Input(shape=(input_dim,))

    x = keras.layers.Dropout(dropout_rate_input)(inputs)

    for i in range(n_layers):
        x = keras.layers.Dense(
            units=hidden_units,
            activation='relu'
            )(x)
        x = keras.layers.Dropout(dropout_rate_hidden)(x)

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
