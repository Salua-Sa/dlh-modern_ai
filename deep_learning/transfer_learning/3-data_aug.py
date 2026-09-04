#!/usr/bin/env python3
"""
This Module creates a Keras Sequential model containing common
image data augmentation operations.
"""
from tensorflow import keras


def build_data_augmentation():
    """
    Create a Keras Sequential model containing common image
    data augmentation operations. This augmentation will be
    applied to training images before they are passed into
    the pretrained CNN.

    The function should:
    - Create a tf.keras.Sequential model
    - Add the following augmentation layers:
    - RandomFlip("horizontal")
    - RandomRotation(0.15)
    - RandomZoom(0.15)
    - RandomContrast(0.1)

    Returns:
        Sequential augmentation model
    """
    data_augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=42),
            keras.layers.RandomRotation(0.15, seed=42),
            keras.layers.RandomZoom(0.15, seed=42),
            keras.layers.RandomContrast(0.1, seed=42)
            ]
        )

    return data_augmentation
