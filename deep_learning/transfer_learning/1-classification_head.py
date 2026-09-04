#!/usr/bin/env python3
"""
This module attaches a custom classification
head to a pretrained feature extractor.
"""
from tensorflow import keras


def add_classification_head(base_model, num_classes):
    """
    Attache a custom classification head to a pretrained
    feature extractor.
    The head should:
    - Take the output of the base model
    - Add a dense layers with 128 filters and relu activation
    - Add a final classification layer

    Args:
        base_model: A Keras Model whose output is a pooled
                    feature vector.
        num_classes: An integer representing the number
                     of output classes.

    Returns:
        Keras Model ready for classification.
    """
    x = base_model.output
    x = keras.layers.Dense(units=128, activation='relu')(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=base_model.input, outputs=outputs)

    return model
