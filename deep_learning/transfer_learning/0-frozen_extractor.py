#!/usr/bin/env python3
"""
This module loads the MobileNetV2 model
and uses it as a feature extractor.
"""
from tensorflow import keras


def build_feature_extractor():
    """
    Loads a pretrained CNN model (e.g., MobileNetV2)
    from Keras applications, removes its classification head,
    and freezes its weights.

    The function should:
    - Load MobileNetV2 with weights="imagenet",
    - input_shape=(224, 224, 3) and without it's classification head
    - Freeze the base model
    - Add a GlobalAveragePooling2D layer on top

    Returns:
        Keras Model that outputs features from input images
        using the frozen base model.
    """
    base_model = keras.applications.MobileNetV2(
        weights="imagenet",
        input_shape=(224, 224, 3),
        include_top=False
        )

    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)

    model = keras.Model(inputs, x)

    return model
