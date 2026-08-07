#!/usr/bin/env python3
"""
This module creates a deep neural network
to perform multi-class classificatio
"""
from tensorflow import keras


def build_deep_model(input_dim, hidden_layers):
    """
    Create a deep neural network to perform multi-class classification.
    Use the Sequential class
    The hidden layers must have:
    - ReLu as an activation function

    Args:
        input_dim: Number of input features.
        hidden_layers: List of integers representing the number
        of neurons in each hidden layer e.g., [16, 8, 4]
        for three hidden layers.

    Returns:
        model: Keras model
    """
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(input_dim,)))

    for unit in hidden_layers:
        model.add(keras.layers.Dense(units=unit, activation="relu"))

    model.add(keras.layers.Dense(10, activation='softmax'))

    return model
