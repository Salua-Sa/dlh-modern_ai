#!/usr/bin/env python3
"""This module trains a CNN model.
"""
from tensorflow import keras


def compile_and_train_cnn(model, epochs, batch_size, x_train, y_train, x_val,
                          y_val, optimizer_name='adam', optimizer_params=None):
    """
    Train a CNN model.

    Args:
        model: The CNN model to be trained.
        epochs: int, the number of training epochs.
        batch_size: int, the size of the batches for training.
        optimizer_name: str, the name of the optimizer to use.
        optimizer_params: dict, additional parameters for the optimizer.

    Returns:
        the trained CNN model, raining history object.
    """
    if optimizer_params is None:
        optimizer_params = {}

    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(**optimizer_params)
    elif optimizer_name == 'sgd':
        optimizer = keras.optimizers.SGD(**optimizer_params)
    elif optimizer_name == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(**optimizer_params)
    else:
        raise ValueError("Invalid optimizer name")

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        verbose=2
                        )

    return model, history
