#!/usr/bin/env python3
"""This module builds the MobileNetV1 architecture.
"""
from tensorflow import keras
mobilenet_backbone = __import__(
    '5-mobilenet_backbone').mobilenet_backbone


def mobilenet(input_shape=(224, 224, 3), num_classes=1000):
    """
    Build the MobileNetV1 architecture as described in MobileNets:
    Efficient Convolutional Neural Networks for Mobile Vision Applications.
    The model should include:
    - An input layer.
    - The MobileNet backbone.
    - A global average pooling layer.
    - A final Dense layer with softmax activation.

    Args:
        input_shape: tuple representing the input image shape.
        num_classes: number of output classes.

    Returns:
         a Keras Model instance representing MobileNetV1.
    """
    inputs = keras.Input(shape=input_shape)

    x = mobilenet_backbone(inputs)

    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(
        num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='mobilenet')

    return model
