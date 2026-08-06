#!/usr/bin/env python3
"""This module creates a shallow neural
network with a single hidden layer.
"""
from tensorflow import keras


def build_model(input_dim, neuros_h):
    """
    Create a shallow neural network with a single
    hidden layer to perform multi-class classification
    using the Sequential class.
    Sigmoid as an activation function for the hidden layer
    Softmax as an activation function for the output layer

    Args:
        input_dim: Number of input features.
        neurons_h: Number of neurons for the hidden layer

    Returns:
    model: keras model.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(units=neuros_h, activation="sigmoid"),
        keras.layers.Dense(10, activation="softmax")
        ])

    return model
