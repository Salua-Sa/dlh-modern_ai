#!/usr/bin/env python3
"""
This module creates a Keras model for multi-class classification, where the
model architecture and training parameters are tuned via Keras tuner.
"""
from tensorflow import keras


def build_model(hp):
    """
    Create a Keras model for multi-class classification,
    where the model architecture and training parameters
    are tuned via Keras tuner.
    The model should include the following tunable aspects:
    - Input Layer:
      - The model will take in input vectors of shape (784,).
    - Hidden Layers:
      - The number of hidden layers and their configurations
        should be tunable:
          - `num_layers`: (int) The number of hidden layers in the network.
          - `units`: (int) The number of neurons in each hidden layer.
          - `activation`: (str) The activation function for each hidden layer.
    - Output Layer:
      - The model should have a Dense output layer with 10 units,
        using the softmax activation function, for multi-class classification.
    - Optimizer and Learning Rate:
      - Use the Adam optimizer.
      - learning_rate: (float) The learning rate for the Adam optimizer,
         selected from one of the fixed values: 1e-2 or 1e-3.

    Args:
        hp: An instance of HyperParameters provided by Keras Tuner that
        defines the search space for the hyperparameters.

    Returns:
        A compiled Keras Sequential model based on the
        hyperparameters defined in the hp object.
    """
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(784,)))

    # The number of hidden layers in the network (between 1 and 2).
    num_layers = hp.Int(
        'num_layers',
        min_value=1,
        max_value=2)

    # The number of neurons (between 4 and 12, with a step size of 4).
    units = hp.Int(
        'units',
        min_value=4,
        max_value=12,
        step=4)

    # The activation function. Choose from `relu` or `sigmoid`.
    activation = hp.Choice(
        'activation',
        values=['relu', 'simoid']
        )

    # The learning rate selected from one of the fixed values: 1e-2 or 1e-3.
    learning_rate = hp.Choice(
        'learning_rate',
        values=[1e-2, 1e-3])

    for i in range(num_layers):
        model.add(
            keras.layers.Dense(
                units=units,
                activation=activation
                )
            )

    model.add(
        keras.layers.Dense(
            units=10,
            activation='softmax')
        )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
        )

    return model
