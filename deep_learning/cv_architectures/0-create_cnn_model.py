#!/usr/bin/env python3
"""This module creates a Convolutional Neural
Network (CNN) model."""
from tensorflow import keras


def create_cnn_model(input_shape, filters,
                     kernel_sizes, activations, pooling_type='max'):
    """
    Creates a Convolutional Neural Network (CNN) model.

    Args:
        input_shape: tuple, the shape of the input data.
        filters: list, the number of filters in each convolutional layer.
        kernel_sizes: list, the size of the kernels.
        activations: list, the activation functions.
        pooling_type: str, the type of pooling
                      ('max' or 'avg', default is 'max').

    Returns:
        compiled CNN model.
    """
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    zip(filters, kernel_sizes, activations)

    for filter, kernel, activation in zip(filters, kernel_sizes, activations):
        model.add(keras.layers.Conv2D(filters=filter,
                                      kernel_size=kernel,
                                      activation=activation
                                      )
                  )
        if pooling_type == "max":
            model.add(keras.layers.MaxPooling2D())
        elif pooling_type == "avg":
            model.add(keras.layers.AveragePooling2D())
        else:
            raise ValueError("Pooling type must be max or avg")

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    return model
